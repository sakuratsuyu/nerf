import torch
import torch.nn.functional as F
import numpy as np

from model import NeRF

from typing import Tuple, Callable

def get_rays(H:int, W:int, K:np.ndarray, c2w:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
        Get the origins and directions of all rays of the image in the world coordinate.


        Args:
            H: int. Height of image in pixels.
            W: int. Width of image in pixels.
            K: numpy.ndarray. size (3, 3). Intrinsic matrix.
            c2w:  torch.Tensor. size (3, 4). Camera to World matrix.
        
        Returns:
            rays_o: torch.Tensor. size (H, W, 3). Coordinates of camera. 
            rays_d: torch.Tensor. size (H, W, 3). Directions of rays.
    '''

    x, y = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing='xy')
    directions = torch.stack([
                                 (x - K[0][2]) / K[0][0],
                                -(y - K[1][2]) / K[1][1],
                                -torch.ones_like(x)
                             ], dim=-1) # (H, W, 3)

    rays_d = torch.sum(directions[:, :, np.newaxis, :] * c2w[:, :3], dim=-1) # equivelent to `rays_d[i, j] = c2w[:, :3] @ directions[i, j]`
    rays_o = c2w[:, 3].expand(rays_d.shape)

    return rays_o, rays_d


def _model_batch(model:NeRF, network_chunk:int) -> Callable:
    '''
        Args:
            model: model.NeRF. Model NeRF.
            network_chunk: int. Number of points sent through network in parallel.

        Returns:
            func: function. It feeds a small chunk of data each time to the model NeRF.
    '''

    def func(pos_embed:torch.Tensor, dirs_embed:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N = pos_embed.shape[0]
        ret = [model(pos_embed[i: i + network_chunk], dirs_embed[i: i + network_chunk]) for i in range(0, N, network_chunk)]
        ret = torch.cat(ret, dim=0)
        return ret[:, :3], ret[:, 3]
    return func


def _propogate(N_rays:int, N_samples:int, pos_embed:torch.Tensor, dirs_embed:torch.Tensor, model:NeRF,
               z_vals:torch.Tensor, rays_d:torch.Tensor, raw_noise_std:float, white_bkgd:bool, network_chunk:int) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
        Forward propogate to get RGB + alpha, then do volume rendering.

        Args:
            N_rays: int. Number of rays.
            N_samples: int. Number of sample points of each ray.
            pos_embed: torch.Tensor. size (N_rays * N_samples, output_dim).
            dirs_embed: torch.Tensor. size (N_rays * N_samples, output_dim).

            model: model.NeRF. Model NeRF.

            z_vals: torch.Tensor. size (N_rays, N_samples). Depths of sample points.
            rays_d: torch.Tensor. size (N_rays, 3). Directions of the rays.

            raw_noise_std: float. Standard deviation of noise (Gaussian) added to regularize sigma output.
            white_bkgd: bool. Whether to render synthetic data on a white background.

            network_chunk: int. Number of points sent through network in parallel.

        Returns:
            colors: torch.Tensor.
            weights: torch.Tensor.
    '''

    rgb, sigma = _model_batch(model, network_chunk)(pos_embed, dirs_embed)
    rgb   =   rgb.reshape(N_rays, N_samples, 3)
    sigma = sigma.reshape(N_rays, N_samples)

    INF = 1e10
    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat([dists, torch.Tensor([INF]).expand(dists[:, :1].shape)], dim=-1)
    dists = dists * torch.norm(rays_d[:, None, :], dim=-1)

    noise = 0
    if raw_noise_std:
        noise = torch.randn(sigma.shape) * raw_noise_std

    alpha = 1.0 - torch.exp(- dists * F.relu(sigma + noise))
    t = torch.cat([torch.ones(alpha.shape[0], 1), 1.0 - alpha + 1e-10], dim=-1) # (N_rays, N_samples + 1)
    T = torch.cumprod(t, dim=-1)[:, :-1] # (N_rays, N_samples)
    weights = alpha * T # (N_rays, N_samples)

    color = torch.sum(weights[:, :, None] * rgb, dim=-2) # (N_rays, 3)
    depth = torch.sum(weights * z_vals, -1)[:, None]
    acc = torch.sum(weights, dim=-1) # (N_rays)

    if white_bkgd:
        color = color + (1 - acc[:, None])

    return color, depth, weights


def _sample_pdf(bins:torch.Tensor, weights:torch.Tensor, N_importance:int, det=False) -> torch.Tensor:
    '''
        Sample more points for the fine model by p.d.f.

        Args:
            bins: torch.Tensor. size (N_rays, N_samples - 1). z_vals_mid.
            weights: torch.Tensor. size (N_rays, N_samples - 2). probability of the ray to stop at each sample point.
            N_importance: int.
            det: bool.

        Returns:
            samples: torch.Tensor.
    '''

    # Get pdf
    weights = weights + 1e-5 # prevent NaNs
    pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples - 2)
    cdf = torch.cumsum(pdf, dim=-1) # (N_rays, N_samples - 2)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # (N_rays, N_samples - 1)

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, steps=N_importance)
        u = u.expand(list(cdf.shape[:-1]) + [N_importance])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_importance])

    # Invert CDF
    u = u.contiguous()
    idx = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(idx - 1), idx - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(idx), idx)
    idx_g = torch.stack([below, above], -1)  # (N_rays, N_importance, 2)

    matched_shape = [idx_g.shape[0], idx_g.shape[1], cdf.shape[-1]]
    # cdf_g[i, j, k] = cdf'[i, j, idx_g[i, j, k]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, idx_g)
    # bins_g[i, j, k] = bins'[i, j, idx_g[i, j, k]]
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, idx_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[...,0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def _render(rays_packed:torch.Tensor, render_args:dict) -> torch.Tensor:
    '''
        Render rays in a chunk.

        Args:
            rays_packed: torch.Tensor. size (2, render_chunk, 3). Ray origin and direction for each example in batch.
            render_args: dict. `train_args` or `test_args`.

        Returns:
            ret: torch.Tensor. size (2, chunk, 3). Predicted RGB values for rays by fine model and coarse model.
    '''

    model         = render_args['network']
    model_fine    = render_args['network_fine']
    near          = render_args['near']
    far           = render_args['far']
    N_importance  = render_args['N_importance']
    N_samples     = render_args['N_samples']
    embedder      = render_args['embedder']
    embedder_dirs = render_args['embedder_dirs']
    perturb       = render_args['perturb']
    raw_noise_std = render_args['raw_noise_std']
    white_bkgd    = render_args['white_bkgd']
    network_chunk = render_args['network_chunk']

    N_rays = rays_packed.shape[1] # N_rays = render_chunk
    rays_o, rays_d = rays_packed

    ## Coarse Network

    # Directions
    dirs = rays_d
    dirs /= torch.norm(dirs, dim=-1, keepdim=True)

    # Position by stratified sampling
    t_vals = torch.linspace(0.0, 1.0, steps=N_samples)
    z_vals = near * (1.0 - t_vals) + far * t_vals
    z_vals = z_vals.expand([N_rays, N_samples]) # (N_rays, N_samples)

    if perturb > 0:
        mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
        lower = torch.cat([z_vals[:, :1] , mids], dim=-1)
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
        t_rand = torch.rand(z_vals.shape)

        z_vals = lower + (upper - lower) * t_rand

    # r = o + td
    pos = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[:, :, None] # (N_rays, N_samples, 3)
    dirs = dirs[:, None].expand(pos.shape) # (N_rays, N_samples, 3)

    # Flatten
    pos = pos.reshape([N_rays * N_samples, 3])
    dirs = dirs.reshape([N_rays * N_samples, 3])

    # Position Embedding
    pos_embed = embedder(pos) # (N_rays * N_samples, embbder.output_dim)
    dirs_embed = embedder_dirs(dirs) # (N_rays * N_samples, embedder_dirs.output_dim)

    # Forward Propogate
    color_0, depth_0, weights = _propogate(N_rays, N_samples, pos_embed, dirs_embed, model,
                                  z_vals, rays_d, raw_noise_std, white_bkgd, network_chunk)

    ## Fine Network

    # Directions
    dirs = rays_d
    dirs /= torch.norm(dirs, dim=-1, keepdim=True)

    # Position by PDF sampling
    z_vals_mid = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
    z_samples = _sample_pdf(z_vals_mid, weights[:, 1:-1], N_importance, det=(perturb == 0.0))
    z_samples = z_samples.detach()
    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)

    # r = o + td
    pos = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[:, :, None] # (N_rays, N_samples + N_importance, 3)
    dirs = dirs[:, None].expand(pos.shape) # (N_rays, N_samples + N_importance, 3)

    # Flatten
    pos = pos.reshape([N_rays * (N_samples + N_importance), 3])
    dirs = dirs.reshape([N_rays * (N_samples + N_importance), 3])

    # Position Embedding
    pos_embed = embedder(pos)
    dirs_embed = embedder_dirs(dirs)

    color, depth, _ = _propogate(N_rays, N_samples + N_importance, pos_embed, dirs_embed, model_fine, 
                          z_vals, rays_d, raw_noise_std, white_bkgd, network_chunk)

    return torch.cat([color, color_0], dim=-1), torch.cat([depth, depth_0], dim=-1)


def render_batch(rays_packed:torch.Tensor, render_args:dict, render_chunk:int) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
        Render rays in multiple batches.

        Args:
            rays_packed: torch.Tensor. size (2, batch_size, 3). Ray origin and direction for each example in batch.
            render_args: dict. `train_args` or `test_args`.

        Returns:
            rgb: torch.Tensor. size (batch_size, 3). Predicted RGB values for rays by fine model.
            rgb_0: torch.Tensor. size (batch_size, 3). Predicted RGB values for rays by coarse model.
    '''

    N_rays = rays_packed.shape[1] # batch_size
    # color, depth = [_render(rays_packed[:, i: i + render_chunk, :], render_args) for i in range(0, N_rays, render_chunk)]
    color = []
    depth = []
    for i in range(0, N_rays, render_chunk):
        c, d = _render(rays_packed[:, i: i + render_chunk, :], render_args)
        color.append(c)
        depth.append(d)
    color = torch.cat(color, dim=0)
    depth = torch.cat(depth, dim=0)
    return color[:, :3], color[:, 3:], depth[:, :1], depth[:, 1:]


def _render_test(H:int, W:int, K:np.ndarray, pose:torch.Tensor, render_args:dict, render_chunk:int) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
        Render an image of given pose.

        Args:
            H: int. Height of the image to render.
            W: int. Width of the image to render.
            K: numpy.ndarray. size (3, 3). Intrinsic matrix.
            pose: torch.Tensor. size (4, 4). Extrinsic matrix.
            render_args: dict. `test_args`.
            render_chunk: int. Number of rays processed in parallel.

        Returns:
            rgb: torch.Tensor. size (H * W, 3). Predicted RGB values for rays by fine model.
            _: torch.Tensor. size (H * W, 3). Predicted RGB values for rays by coarse model. But we don't care it when rendering the final video.
    '''

    rays_o, rays_d = get_rays(H, W, K, pose[:3, :4])
    rays_o = torch.reshape(rays_o, [-1, 3])
    rays_d = torch.reshape(rays_d, [-1, 3])
    rays_packed = torch.stack([rays_o, rays_d], dim=0)
    return render_batch(rays_packed, render_args, render_chunk)


def render_path(render_poses:torch.Tensor, hwf:list, K:np.ndarray, render_args:dict, render_chunk:int) -> np.ndarray:
    '''
        Render the result or test data.

        Args:
            render_poses: numpy.ndarray. size (N, 3, 4). Camera poses where the images are needed to render.
            hwf: list. [height, width, focal length].
            K: numpy.ndarray. size (3, 3). Intrinsic matrix.
            render_args: dict. `test_args`.

        Returns:
            rgbs: numpy.ndarray. size (H * W, 3). Predicted RGB values for rays by fine model.
    '''

    from tqdm import tqdm

    H, W, _ = hwf

    rgbs = []
    depths = []

    for _, pose in enumerate(tqdm(render_poses)):
        rgb, _, depth, _ = _render_test(H, W, K, pose, render_args, render_chunk)
        rgb = rgb.reshape(H, W, 3)
        rgbs.append(rgb.cpu().numpy())
        depth = depth.reshape(H, W, 1)
        depths.append(depth.cpu().numpy())

    rgbs = np.stack(rgbs, axis=0)
    depths = np.stack(depths, axis=0)

    return rgbs, depths
