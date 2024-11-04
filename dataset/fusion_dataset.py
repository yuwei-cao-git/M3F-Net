import torch
from torch.utils.data import Dataset
import numpy as np
import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split


class SuperpixelDataset(Dataset):
    def __init__(self, combined_data_dir, transform=None):
        self.combined_data_files = [
            os.path.join(combined_data_dir, f)
            for f in os.listdir(combined_data_dir)
            if f.endswith(".npz")
        ]
        self.transform = transform

        # Optional: Shuffle or sort file paths if needed
        self.file_paths.sort()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.combined_data_files[idx], allow_pickle=True)
        tile_image = data["tile_image"]
        label_array = data["label_array"]
        superpixel_mask = data["superpixel_mask"]
        nodata_mask = data["nodata_mask"]
        polyid_to_point_cloud = data["polyid_to_point_cloud"].item()
        polyid_to_label = data["polyid_to_label"].item()

        # Process data as needed for your model
        # For example, you might aggregate point clouds or extract features

        sample = {
            "tile_image": tile_image,
            "label_array": label_array,
            "superpixel_mask": superpixel_mask,
            "nodata_mask": nodata_mask,
            "polyid_to_point_cloud": polyid_to_point_cloud,
            "polyid_to_label": polyid_to_label,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class SuperpixelDataModule(LightningDataModule):
    def __init__(self, data_dir, point_cloud_dir, batch_size=4, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.point_cloud_dir = point_cloud_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Called on every GPU
        full_dataset = SuperpixelDataset(self.data_dir, self.point_cloud_dir)

        # Split dataset into train, val, test
        train_size = int(0.8 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        # Implement custom collation to handle variable-size data
        batch_size = len(batch)

        # Collate tile images and labels (assuming same shape)
        tile_images = torch.stack([sample["tile_image"] for sample in batch])
        label_arrays = torch.stack([sample["label_array"] for sample in batch])
        superpixel_masks = torch.stack([sample["superpixel_mask"] for sample in batch])
        nodata_masks = torch.stack([sample["nodata_mask"] for sample in batch])

        # Collate point clouds and polyids
        point_clouds_list = [sample["point_clouds"] for sample in batch]
        polyids_list = [sample["polyids"] for sample in batch]

        # Since point clouds may have different numbers of superpixels, we handle them as lists
        batch = {
            "tile_image": tile_images,  # (batch_size, bands, H, W)
            "label_array": label_arrays,  # (batch_size, classes, H, W)
            "superpixel_mask": superpixel_masks,  # (batch_size, H, W)
            "nodata_mask": nodata_masks,  # (batch_size, H, W)
            "point_clouds": point_clouds_list,  # List of tensors
            "polyids": polyids_list,  # List of tensors
        }

        return batch
