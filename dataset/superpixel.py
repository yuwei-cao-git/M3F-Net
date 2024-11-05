import torch
from torch.utils.data import Dataset
import numpy as np
import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from os.path import join
import torchvision.transforms.v2 as transforms


class SuperpixelDataset(Dataset):
    def __init__(
        self, superpixel_data_dir, image_transform=None, point_cloud_transform=None
    ):
        self.superpixel_files = [
            os.path.join(superpixel_data_dir, f)
            for f in os.listdir(superpixel_data_dir)
            if f.endswith(".npz")
        ]
        self.image_transform = image_transform
        self.point_cloud_transform = point_cloud_transform

    def __len__(self):
        return len(self.superpixel_files)

    def __getitem__(self, idx):
        data = np.load(self.superpixel_files[idx], allow_pickle=True)
        superpixel_images = data[
            "superpixel_images"
        ]  # List of images from different seasons
        coords = data["point_cloud"]
        xyz = coords - np.mean(coords, axis=0)
        label = data["label"]

        # Apply transforms if needed
        if self.image_transform:
            if self.aug == "random":
                self.transform = transforms.RandomApply(
                    torch.nn.ModuleList(
                        [
                            transforms.RandomCrop(size=(128, 128)),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.ToDtype(torch.float32, scale=True),
                            transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                            transforms.RandomRotation(degrees=(0, 180)),
                            transforms.RandomAffine(
                                degrees=(30, 70),
                                translate=(0.1, 0.3),
                                scale=(0.5, 0.75),
                            ),
                        ]
                    ),
                    p=0.3,
                )
            else:
                self.transform = transforms.Compose(
                    [
                        transforms.RandomCrop(size=(128, 128)),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToDtype(torch.float32, scale=True),
                        transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                        transforms.RandomRotation(degrees=(0, 180)),
                        transforms.RandomAffine(
                            degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)
                        ),
                    ]
                )
            superpixel_images = [self.transform(img) for img in superpixel_images]
        # Apply point cloud transforms if any
        if self.point_cloud_transform:
            coords, xyz, label = self.point_cloud_transform(xyz, coords, label)
        # After applying transforms
        superpixel_images = [torch.from_numpy(img) for img in superpixel_images]
        pc_feat = torch.from_numpy(coords).float()
        xyz = torch.from_numpy(xyz).float()
        label = torch.tensor(label, dtype=torch.float32)

        sample = {
            "images": superpixel_images,
            "point_cloud": xyz,
            "pc_feat": pc_feat,
            "label": label,
        }
        return sample


class SuperpixelDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config["batch_size"]
        self.image_transform = config["imag_transforms"]
        self.point_cloud_transform = config["pc_transforms"]

        self.processed_dir = join(config["data_dir"], f'{self.config["resolution"]}m')
        self.data_dir = {
            "train": join(self.processed_dir, "train/superpixel"),
            "val": join(self.processed_dir, "val/superpixel"),
            "test": join(self.processed_dir, "test/superpixel"),
        }

    def setup(self, stage=None):
        # Create datasets for train, validation, and test
        self.train_dataset = SuperpixelDataset(
            self.data_dir["train"],
            transform=self.image_transform,
            point_cloud_transform=self.point_cloud_transform,
        )
        self.val_dataset = SuperpixelDataset(
            self.data_dir["val"], image_transform=None, point_cloud_transform=None
        )
        self.test_dataset = SuperpixelDataset(
            self.data_dir["test"], image_transform=None, point_cloud_transform=None
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config["num_workers"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
        )
