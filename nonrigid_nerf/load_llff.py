import numpy as np
import os
import imageio

def load_llff_data(basedir, factor=8):
    """
    Function for loading LLFF data.
    This function loads images, poses, bounds, and render poses from the LLFF dataset.

    Args:
        basedir (str): Base directory containing the LLFF data.
        factor (int): Downsampling factor for the images.

    Returns:
        images (np.ndarray): Array of images.
        poses (np.ndarray): Array of camera poses.
        bds (np.ndarray): Array of bounds.
        render_poses (np.ndarray): Array of render poses.
    """
    # Load images
    image_dir = os.path.join(basedir, 'images')
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.JPG')])
    images = np.array([imageio.imread(f) for f in image_files])

    # Load poses
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])
    bds = poses_arr[:, -2:]

    # Load render poses
    render_poses = np.load(os.path.join(basedir, 'render_poses.npy'))

    return images, poses, bds, render_poses

def load_intrinsics(scene_dir):
    """
    Function for loading camera intrinsics.
    This function loads the camera intrinsics from a specified directory.

    Args:
        scene_dir (str): Directory containing the scene data.

    Returns:
        intrinsics (dict): Dictionary containing camera intrinsics.
    """
    intrinsics_file = os.path.join(scene_dir, 'intrinsics.txt')
    if not os.path.exists(intrinsics_file):
        raise FileNotFoundError(f"Intrinsics file not found in {scene_dir}")

    intrinsics = {}
    with open(intrinsics_file, 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            intrinsics[key] = float(value)

    return intrinsics
