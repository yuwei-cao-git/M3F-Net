# import glob
import os
import random
import numpy as np
import pandas as pd
import torch
from .common import read_las
from torch.utils.data import Dataset


def rotate_points(coords, x=None):
    rotation = np.random.uniform(-180, 180)
    # Convert rotation values to radians
    rotation = np.radians(rotation)

    # Rotate point cloud
    rot_mat = np.array(
        [
            [np.cos(rotation), -np.sin(rotation), 0],
            [np.sin(rotation), np.cos(rotation), 0],
            [0, 0, 1],
        ]
    )
    aug_coords = coords
    aug_coords[:, :3] = np.matmul(aug_coords[:, :3], rot_mat)
    if x is None:
        aug_x = None
    else:
        aug_x = x
        aug_x[:, :3] = np.matmul(aug_x[:, :3], rot_mat)

    return aug_coords, aug_x


def point_removal(coords, n, x=None):
    # Get list of ids
    idx = list(range(np.shape(coords)[0]))
    random.shuffle(idx)  # shuffle ids
    idx = np.random.choice(
        idx, n, replace=False
    )  # pick points randomly removing up to 10% of points

    # Remove random values
    aug_coords = coords[idx, :]  # remove coords
    if x is None:  # remove x
        aug_x = None
    else:
        aug_x = x[idx, :]

    return aug_coords, aug_x


def random_noise(coords, n, dim=1, x=None):
    # Random standard deviation value
    random_noise_sd = np.random.uniform(0.01, 0.025)

    # Add/Subtract noise
    if np.random.uniform(0, 1) >= 0.5:  # 50% chance to add
        aug_coords = coords + np.random.normal(
            0, random_noise_sd, size=(np.shape(coords)[0], 3)
        )
        if x is None:
            aug_x = None
        else:
            aug_x = x + np.random.normal(
                0, random_noise_sd, size=(np.shape(x)[0], dim)
            )  # added [0] and dim
    else:  # 50% chance to subtract
        aug_coords = coords - np.random.normal(
            0, random_noise_sd, size=(np.shape(coords)[0], 3)
        )
        if x is None:
            aug_x = None
        else:
            aug_x = x - np.random.normal(
                0, random_noise_sd, size=(np.shape(x)[0], dim)
            )  # added [0] and dim

    # Randomly choose up to 10% of augmented noise points
    use_idx = np.random.choice(
        aug_coords.shape[0],
        n,
        replace=False,
    )
    aug_coords = aug_coords[use_idx, :]  # get random points
    aug_coords = np.append(coords, aug_coords, axis=0)  # add points
    if x is None:
        aug_x = None
    else:
        aug_x = aug_x[use_idx, :]  # get random point values
        aug_x = np.append(x, aug_x, axis=0)  # add random point values # ADDED axis=0

    return aug_coords, aug_x


class AugmentPointCloudsInPickle(Dataset):
    """Point cloud dataset where one data point is a file."""

    def __init__(
        self,
        filepath,
        pickle,
    ):
        self.filepath = filepath
        self.pickle = pd.read_pickle(pickle)
        super().__init__()

    def __len__(self):
        return len(self.pickle)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get Filename
        pickle_idx = self.pickle.iloc[idx : idx + 1]
        filename = pickle_idx["FileName"].item()

        # Get file path
        file = os.path.join(self.filepath, filename)

        # Read las/laz file
        coords = read_las(file, get_attributes=False)

        xyz = coords - np.mean(coords, axis=0)  # centralize coordinates

        # Augmentation
        n = random.randint(round(len(xyz) * 0.9), len(xyz))
        aug_xyz, aug_coords = point_removal(xyz, n, x=coords)
        aug_xyz, aug_coords = random_noise(aug_xyz, n=(len(xyz) - n), x=aug_coords)
        xyz, coords = rotate_points(aug_xyz, x=aug_coords)

        # Get Target
        target = pickle_idx["perc_specs"].item()
        target = target.replace("[", "")
        target = target.replace("]", "")
        target = target.split(",")
        target = [float(i) for i in target]  # convert items in target to float

        coords = torch.from_numpy(coords).float()
        xyz = torch.from_numpy(xyz).float()
        target = torch.from_numpy(np.array(target)).type(torch.FloatTensor)
        if xyz.shape[0] < 100:
            return None
        return coords, xyz, target


class PointCloudTransform:
    def __init__(self, min_percentage=0.9, sigma=0.01, clip=0.05):
        self.min_percentage = min_percentage
        self.sigma = sigma
        self.clip = clip

    def __call__(self, xyz, coords, target):
        # Point Removal
        n = random.randint(round(len(xyz) * 0.9), len(xyz))
        aug_xyz, aug_coords = point_removal(xyz, n, x=coords)
        aug_xyz, aug_coords = random_noise(aug_xyz, n=(len(xyz) - n), x=aug_coords)
        xyz, coords = rotate_points(aug_xyz, x=aug_coords)

        # Get Target
        target = target.replace("[", "")
        target = target.replace("]", "")
        target = target.split(",")
        target = [float(i) for i in target]  # convert items in target to float

        return coords, xyz, target
