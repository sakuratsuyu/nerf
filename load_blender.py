import os, json
import imageio
import numpy as np
import torch
import cv2

def _pose_spherical(radius:float, theta:float, phi:float):
    '''
        Args:
            radius, theta, phi: float. spherical coordinates.

        Returns:
            c2w: <class 'tensor'>. corresponding Camera to world matrix.
    '''
    trans_radius = lambda r : torch.Tensor([
                                               [1, 0, 0, 0],
                                               [0, 1, 0, 0],
                                               [0, 0, 1, r],
                                               [0, 0, 0, 1],
                                           ]).float()
    rot_phi = lambda phi : torch.Tensor([
                                          [1, 0, 0, 0],
                                          [0, np.cos(phi), -np.sin(phi), 0],
                                          [0, np.sin(phi),  np.cos(phi), 0],
                                          [0, 0, 0, 1],
                                      ]).float()
    rot_theta = lambda theta : torch.Tensor([
                                          [np.cos(theta), 0, -np.sin(theta), 0],
                                          [0, 1, 0, 0],
                                          [np.sin(theta), 0,  np.cos(theta), 0],
                                          [0, 0, 0, 1],
                                      ]).float()

    c2w = trans_radius(radius)
    c2w = rot_phi(phi / 180 * np.pi) @ c2w
    c2w = rot_theta(theta / 180 * np.pi) @ c2w
    rota = torch.Tensor(np.array([
                        [-1, 0, 0, 0],
                        [ 0, 0, 1, 0],
                        [ 0, 1, 0, 0],
                        [ 0, 0, 0, 1],
                    ]))
    c2w = rota @ c2w
    return c2w

def load_blender_data(basedir:str, half_res=False):
    '''
        Load Data with blender format.

        Args:
            basedir: string. path of the data directory.
            half_res: boolean. whether to downsample (half resolution).

        Returns:
            imgs: <class 'numpy.ndarray'>. size (N, H, W, C). NOTE that the images have 4 channels (RGBA).
            poses: <class 'numpy.ndarray'>. size (N, 4, 4). poses of each imgs.
                
                NOTE: `poses[:, :, i]` of each image is of the form
                    - R is a 3 by 3 roation matrix.
                    - t is the translation vector.
                    - H, W, f are the height, width, and focal length.

                 │      0  1  2   3
                 │     ┌───────┐┌───┐
                 │  0  │       ││   │
                 │  1  │   R   ││ t │
                 │  2  │       ││   │
                 │     └───────┘└───┘
                 │  3   0  0  0   1

            rendered_poses: <class 'numpy.ndarray'>. size (40, 4, 4). poses to render, in the same form of extrinsic matrix.
            indices: <class 'list'>, a list of list. size (idx_train, idx_val, idx_test). each element is a list of indices of images.
    '''

    splits = ['train', 'val', 'test']
    dataset = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as file:
            dataset[s] = json.load(file)

    imgs_all = []
    poses_all = []
    counts = [0]
    for s in splits:
        data = dataset[s]
        imgs = []
        poses = []

        for frame in data['frames']:
            file = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(file))
            poses.append(np.array(frame['transform_matrix']))

        # Normalize
        imgs = (np.array(imgs) / 255.0).astype(np.float32) # [N, H, W, C]
        poses = np.array(poses).astype(np.float32) # [N, 4, 4]
        imgs_all.append(imgs)
        poses_all.append(poses)
        counts.append(counts[-1] + imgs.shape[0])

    indices = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(imgs_all, axis=0)
    poses = np.concatenate(poses_all, axis=0)

    H = int(imgs[0].shape[0])
    W = int(imgs[0].shape[1])
    camera_angle_x = dataset[splits[0]]['camera_angle_x']
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    # Generate the poses to render. The path is a sphere around the object. radius and phi are fixed, while theta = `angle`
    radius = 4.0
    phi = -30.0
    render_poses = [_pose_spherical(radius, theta, phi) for theta in np.linspace(-180, 180, 40 + 1)[:-1]]
    render_poses = torch.stack(render_poses, dim=0)

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.0

        N = imgs.shape[0]
        C = 4
        imgs_half = np.zeros([N, H, W, C])
        for i, img in enumerate(imgs):
            imgs_half[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half

    return [H, W, focal], indices, imgs, poses, render_poses
