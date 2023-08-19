import os
import imageio
import configargparse

import torch
import numpy as np

from tqdm import tqdm, trange

from load_blender import load_blender_data
from model import NeRF, Embedder
from render import get_rays, render_batch, render_path

from typing import Tuple
from torch.optim import Optimizer

torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]))
to8b = lambda x : (255 * np.clip(x, 0, 1)).astype(np.uint8)

def create_nerf(args) -> Tuple[dict, dict, int, Optimizer]:
    '''
        Initiate NeRF Model.

        Args:
            args: argparse.Namespace. Args of NeRF model.
        
        Returns:
            train_args: dict. Args to train the model.
            test_args: dict. Args to test the model.
            start:  int. Start step.
            optimizer: torch.optim.adam.Adam. Adam optimizer.
    '''

    # Initiate Embedder
    input_dim = 3
    include_input = True

    num_freq = args.num_freq
    max_freq_log = num_freq - 1
    embedder = Embedder(input_dim, max_freq_log, num_freq, include_input)
    num_freq_views = args.num_freq_views
    max_freq_log_views = num_freq_views - 1
    embedder_dirs = Embedder(input_dim, max_freq_log_views, num_freq_views, include_input)

    input_ch = embedder.output_dim
    input_ch_views = embedder_dirs.output_dim
    skips = [4]

    # Coarse Model
    model = NeRF(layer=args.layer, channel=args.channel, skips = skips,
                 input_ch = input_ch, input_ch_views = input_ch_views)
    model = model.to(device)
    params = list(model.parameters())

    # Fine Model
    model_fine = NeRF(layer=args.layer, channel=args.channel, skips = skips,
                 input_ch = input_ch, input_ch_views = input_ch_views)
    model_fine = model_fine.to(device)
    params += list(model_fine.parameters())


    # Return values

    optimizer = torch.optim.Adam(params=params, lr=args.lr, betas=(0.9, 0.999))

    start = 0

    ##########################

    # Load checkpoints
    basedir = args.basedir
    expname = args.expname
    ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    # `no_reload` means that do not reload weights from saved ckpt
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        model.load_state_dict(ckpt['network_state_dict'])
        model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    train_args = {
        'network': model,
        'network_fine': model_fine,
        'near': args.near,
        'far': args.far,
        'N_samples': args.N_samples,
        'N_importance': args.N_importance,
        'embedder': embedder,
        'embedder_dirs': embedder_dirs,
        'perturb': args.perturb,
        'raw_noise_std': args.raw_noise_std,
        'white_bkgd': args.white_bkgd,
        'network_chunk': args.network_chunk
    }

    test_args = {
        'network': model,
        'network_fine': model_fine,
        'near': args.near,
        'far': args.far,
        'N_samples': args.N_samples,
        'N_importance': args.N_importance,
        'embedder': embedder,
        'embedder_dirs': embedder_dirs,
        'perturb': 0.0,
        'raw_noise_std': 0.0,
        'white_bkgd': args.white_bkgd,
        'network_chunk': args.network_chunk
    }

    return train_args, test_args, start, optimizer


def config_parser() -> configargparse.ArgumentParser:
    '''
        Deal with configs.

        Returns:
            parser. configargparse.ArgumentParser.
    '''

    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/nerf_synthetic/lego',
                        help='input data directory')

    # Training Options
    ## Network structure
    parser.add_argument("--layer", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--channel", type=int, default=256,
                        help='channels per layer')
    
    # Hyperparameters: Batch size and Learning rate
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lr", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lr_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')

    ## Parallel
    parser.add_argument("--render_chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--network_chunk", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')

    ## Load Model
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')

    # Rendering Options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')

    parser.add_argument("--near", type=float, default=2.0,
                        help='near bound')
    parser.add_argument("--far", type=float, default=6.0,
                        help='far bound')

    parser.add_argument("--num_freq", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--num_freq_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')

    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--raw_noise_std", type=float, default=0.0,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')

    # Training Options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=0.5, help='fraction of img taken for central crops')

    # Dataset Options
    parser.add_argument("--dataset_type", type=str, default='blender',
                        help='options: llff / blender / deepvoxels')

    ## Blender Flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # Logging/Saving Options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000,
                        help='frequency of render_poses video saving')

    return parser


def train():
    # Load config
    parser = config_parser()
    args = parser.parse_args()

    # Load Data
    hwf, indices, imgs, poses, render_poses = load_blender_data(args.datadir, args.half_res)
    idx_train, idx_val, idx_test = indices

    poses = torch.Tensor(poses).to(device)
    render_poses = torch.Tensor(render_poses).to(device)

    if args.white_bkgd:
        imgs = imgs[..., :3] * imgs[..., -1:] + (1.0 - imgs[..., -1:])
    else:
        imgs = imgs[..., :3]

    print('Loaded blender', hwf, imgs.shape, args.datadir)

    # Height, Width and Focal Length
    H, W, focal = hwf
    # Intrinsic Matrix
    K = np.array([
                     [focal, 0, 0.5 * W],
                     [0, focal, 0.5 * H],
                     [0, 0, 1]
                 ])

    # Save Path
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)

    # Create Nerf Model
    train_args, test_args, start, optimizer = create_nerf(args)

    if args.render_test:
        render_poses = poses[idx_test]

    if args.render_only:
        print('Render Only')
        with torch.no_grad():
            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)

            rgbs = render_path(render_poses, hwf, K, test_args, args.render_chunk)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Training

    N_iters = 200000 + 1
    print('Begin')

    global_step = start
    for i in trange(start + 1, N_iters):

        idx_img = np.random.choice(idx_train)
        target = torch.Tensor(imgs[idx_img]).to(device)
        pose = poses[idx_img, :3, :4]


        rays_o, rays_d = get_rays(H, W, K, pose)

        if i < args.precrop_iters:
            h = int(H // 2 * args.precrop_frac)
            w = int(W // 2 * args.precrop_frac)
            coords = torch.meshgrid(
                torch.linspace(H // 2 - h, H // 2 + h - 1, 2 * h),
                torch.linspace(W // 2 - w, W // 2 + w - 1, 2 * w),
                indexing='ij'
            )
            coords = torch.stack(coords, dim=-1) # (2 * h, 2 * w, 2)
        else:
            coords = torch.meshgrid(
                torch.linspace(0, H - 1, H), 
                torch.linspace(0, W - 1, W),
                indexing='ij'
            )
            coords = torch.stack(coords, dim=-1) # (H, W, 2)

        coords = coords.reshape([-1, 2]) # (H * W, 2) or (2 * h * 2 * w, 2)

        N_rand = args.N_rand
        idx_coords = np.random.choice(coords.shape[0], size=N_rand, replace=False)
        coords_selected = coords[idx_coords].long()

        rays_o = rays_o[coords_selected[:, 0], coords_selected[:, 1]] # (N_rand, 3)
        rays_d = rays_d[coords_selected[:, 0], coords_selected[:, 1]] # (N_rand, 3)
        rays_packed = torch.stack([rays_o, rays_d], dim=0)            # (2, N_rand, 3)
        target = target[coords_selected[:, 0], coords_selected[:, 1]] # (N_rand, 3)

        rgb, rgb0, _, _ = render_batch(rays_packed, train_args, args.render_chunk)

        # Loss
        optimizer.zero_grad()
        img_loss = img2mse(rgb, target)
        psnr = mse2psnr(img_loss)
        loss = img_loss

        img_loss0 = img2mse(rgb0, target)
        # psnr0 = mse2psnr(img_loss0)
        loss += img_loss0

        # Backward Propogation
        loss.backward()
        optimizer.step()

        # Update Learning Rate
        decay_rate = 0.1
        decay_steps = args.lr_decay * 1000
        new_lr = args.lr * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


        ## Refresh the progress bar
        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        ## Test
        if i % args.i_testset == 0:
            pass

        ## Render the video
        if i % args.i_video == 0:
            with torch.no_grad():
                rgbs, depths = render_path(render_poses, hwf, K, test_args, args.render_chunk)
            print('Done, saving', rgbs.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'depth.mp4', to8b(depths / np.max(depths)), fps=30, quality=8)

        ## Save the model
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'optimizer_state_dict': optimizer.state_dict(),
                'network_state_dict': train_args['network'].state_dict(),
                'network_fine_state_dict': train_args['network_fine'].state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        global_step += 1


if __name__=='__main__':
    train()

