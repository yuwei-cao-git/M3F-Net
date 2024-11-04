import torch
from torch.utils.data import Dataset
import numpy as np
import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from os.path import join


class SuperpixelDataset(Dataset):
    def __init__(self, superpixel_data_dir, transform=None):
        self.superpixel_files = [
            os.path.join(superpixel_data_dir, f)
            for f in os.listdir(superpixel_data_dir)
            if f.endswith(".npz")
        ]
        self.transform = transform

    def __len__(self):
        return len(self.superpixel_files)

    def __getitem__(self, idx):
        data = np.load(self.superpixel_files[idx], allow_pickle=True)
        superpixel_images = data[
            "superpixel_images"
        ]  # List of images from different seasons
        point_cloud = data["point_cloud"]
        label = data["label"]

        # Apply transforms if needed
        if self.transform:
            superpixel_images = [self.transform(img) for img in superpixel_images]
            # Apply point cloud transforms if any

        sample = {
            "images": superpixel_images,
            "point_cloud": point_cloud,
            "label": label,
        }
        return sample


class SuperpixelDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.processed_dir = join(config["data_dir"], f'{self.config["resolution"]}m')
        self.data_dir = {
            "train": join(self.processed_dir, "train/superpixel.txt"),
            "val": join(self.processed_dir, "val/superpixel.txt"),
            "test": join(self.processed_dir, "test/superpixel.txt"),
        }

    def setup(self, stage=None):
        # Create datasets for train, validation, and test
        self.train_dataset = SuperpixelDataset(self.data_dir["train"])
        self.val_dataset = SuperpixelDataset(self.data_dir["val"])
        self.test_dataset = SuperpixelDataset(self.data_dir["test"])

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
