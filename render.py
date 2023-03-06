import torch
import torch.nn.functional as F
import numpy as np

from model import NeRF

def get_rays(H:int, W:int, K:np.ndarray, c2w:torch.Tensor):
    '''
        Get the origins and directions of all rays of the image in the world coordinate.


        Args:
            H: int. Height of image in pixels.
            W: int. Width of image in pixels.
            K: <class 'numpy.ndarray'>. size (3, 3). Intrinsic matrix.
            c2w:  <class 'torch.Tensor'>. size (3, 4). Camera to World matrix.
        
        Returns:
            rays_o:  <class 'torch.Tensor'>. size (H, W, 3). coordinates of camera. 
            rays_d:  <class 'torch.Tensor'>. size (H, W, 3). directions of rays.
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


def _model_batch(model:NeRF, network_chunk:int):
    def func(pos_embed, dirs_embed):
        N = pos_embed.shape[0]
        ret = [model(pos_embed[i: i + network_chunk], dirs_embed[i: i + network_chunk]) for i in range(0, N, network_chunk)]
        ret = torch.cat(ret, dim=0)
        return ret[:, :3], ret[:, 3]
    return func


def _propogate(N_rays:int, N_samples:int, pos_embed:torch.Tensor, dirs_embed:torch.Tensor, model:NeRF,
               z_vals:torch.Tensor, rays_d:torch.Tensor, raw_noise_std:float, white_bkgd:bool, network_chunk:int):
    '''
        Forward propogate to get RGB + alpha, then do volume rendering.
    '''

    rgb, alpha = _model_batch(model, network_chunk)(pos_embed, dirs_embed)
    rgb   =   rgb.reshape(N_rays, N_samples, 3)
    alpha = alpha.reshape(N_rays, N_samples)

    INF = 1e10
    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat([dists, torch.Tensor([INF]).expand(dists[:, :1].shape)], dim=-1)
    dists = dists * torch.norm(rays_d[:, None, :], dim=-1)

    noise = 0
    if raw_noise_std:
        noise = torch.randn(alpha.shape) * raw_noise_std

    sigma = 1.0 - torch.exp(- dists * F.relu(alpha + noise))
    t = torch.cat([torch.ones(sigma.shape[0], 1), 1.0 - sigma + 1e-10], dim=-1) # (N_rays, N_samples + 1)
    T = torch.cumprod(t, dim=-1)[:, :-1] # (N_rays, N_samples)
    weights = sigma * T # (N_rays, N_samples)

    color = torch.sum(weights[:, :, None] * rgb, dim=-2) # (N_rays, 3)
    acc = torch.sum(weights, dim=-1)

    if white_bkgd:
        color = color + (1 - acc[:, None])

    return color, weights


def _sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples


def _render(rays_packed:torch.Tensor, render_args:dict):
    '''
        Render rays in a chunk.

        Args:
            rays_packed: <class 'torch.Tensor'>. size [2, batch_size, 3]. Ray origin and direction for each example in batch.
            render_args: `train_args` or `test_args`.

        Returns:
            ret: <class 'torch.Tensor'>. size [2, chunk, 3]. Predicted RGB values for rays by fine model and coarse model.
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

    N_rays = rays_packed.shape[1]
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
    pos_embed = embedder(pos)
    dirs_embed = embedder_dirs(dirs)

    # Forward Propogate
    color_0, weights = _propogate(N_rays, N_samples, pos_embed, dirs_embed, model,
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

    color, _ = _propogate(N_rays, N_samples + N_importance, pos_embed, dirs_embed, model_fine, 
                          z_vals, rays_d, raw_noise_std, white_bkgd, network_chunk)

    return torch.cat([color, color_0], dim=-1)


def render_batch(rays_packed:torch.Tensor, render_args:dict, render_chunk:int):
    '''
        Render rays

        Args:
            rays_packed: <class 'torch.Tensor'>. size [2, batch_size, 3]. Ray origin and direction for each example in batch.
            render_args: `train_args` or `test_args`.

        Returns:
            rgb: [batch_size, 3]. Predicted RGB values for rays by fine model.
            rgb_0: [batch_size, 3]. Predicted RGB values for rays by coarse model.
    '''

    N_rays = rays_packed.shape[1]
    ret = [_render(rays_packed[:, i: i + render_chunk, :], render_args) for i in range(0, N_rays, render_chunk)]
    ret = torch.cat(ret, dim=0)
    return ret[:, :3], ret[:, 3:]


def _render_test(H:int, W:int, K:np.ndarray, pose:torch.Tensor, render_args:dict, render_chunk:int):
    rays_o, rays_d = get_rays(H, W, K, pose[:3, :4])
    rays_o = torch.reshape(rays_o, [-1, 3])
    rays_d = torch.reshape(rays_d, [-1, 3])
    rays_packed = torch.stack([rays_o, rays_d], dim=0)
    return render_batch(rays_packed, render_args, render_chunk)


def render_path(render_poses:torch.Tensor, hwf:list, K:np.ndarray, render_args:dict, render_chunk:int):
    '''
        Render the result or test data.

        Args:
            render_poses: <class 'numpy.ndarray'>. size (N, 3, 4). the camera poses where the images are needed to render.
            hwf: <class 'list'>. [height, width, focal length].
            K: <class 'numpy.ndarray'> [3, 3]. the intrinsic matrix.
            render_args: dict. args for render process.

        Returns:
            rgbs: <class 'list'>. size [H * W, 3].
    '''

    from tqdm import tqdm

    H, W, _ = hwf

    rgbs = []

    for _, pose in enumerate(tqdm(render_poses)):
        rgb, _ = _render_test(H, W, K, pose, render_args, render_chunk)
        rgb = rgb.reshape(H, W, 3)
        rgbs.append(rgb.cpu().numpy())

    rgbs = np.stack(rgbs, axis=0)

    return rgbs
