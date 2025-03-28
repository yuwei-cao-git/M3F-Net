import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from os.path import join
from data_utils.augment import pointCloudTransform, image_transform


class SuperpixelDataset(Dataset):
    def __init__(
        self,
        superpixel_files,
        rotate=None,
        normalization=None,
        image_transform=None,
        point_cloud_transform=None,
    ):
        self.superpixel_files = superpixel_files
        self.image_transform = image_transform
        self.point_cloud_transform = point_cloud_transform
        self.rotate = rotate
        self.norm = normalization

    def __len__(self):
        return len(self.superpixel_files)

    def __getitem__(self, idx):
        data = np.load(self.superpixel_files[idx], allow_pickle=True)
        # Load data from the .npz file
        superpixel_images = data[
            "superpixel_images"
        ]  # Shape: (num_seasons, num_channels, 128, 128)
        coords = data["point_cloud"]  # Shape: (7168, 3)
        label = data["label"]  # Shape: (num_classes,)
        per_pixel_labels = data["per_pixel_labels"]  # Shape: (num_classes, 128, 128)
        nodata_mask = data["nodata_mask"]  # Shape: (128, 128)
        xyz = coords - np.mean(coords, axis=0)

        superpixel_images = torch.from_numpy(
            superpixel_images
        ).float()  # Shape: (num_seasons, num_channels, 128, 128)
        per_pixel_labels = torch.from_numpy(
            per_pixel_labels
        ).float()  # Shape: (num_classes, 128, 128)
        nodata_mask = torch.from_numpy(nodata_mask).bool()

        # Apply transforms if needed
        if self.image_transform != None:
            superpixel_images = image_transform(superpixel_images, self.image_transform)

        # Apply point cloud transforms if any
        if self.point_cloud_transform:
            xyz, coords, label = pointCloudTransform(
                xyz, pc_feat=coords, target=label, rot=self.rotate
            )

        if self.norm:
            m = np.max(np.linalg.norm(xyz, axis=1, keepdims=True))
            norm_xyz = xyz / m
            coords = np.concatenate([coords, norm_xyz], axis=-1)

        # After applying transforms
        coords = torch.from_numpy(coords).float()  # Shape: (7168, 3)
        xyz = torch.from_numpy(xyz).float()  # Shape: (7168, 3)
        label = torch.from_numpy(label).float()  # Shape: (num_classes,)

        sample = {
            "images": superpixel_images,  # Padded images of shape [num_seasons, num_channels, 128, 128]
            "nodata_mask": nodata_mask,  # Padded masks of shape [num_seasons, 128, 128]
            "per_pixel_labels": per_pixel_labels,  # Tensor: (num_classes, 128, 128)
            "point_cloud": xyz,
            "pc_feat": coords,
            "label": label,
        }
        return sample


class SuperpixelDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.image_transform = (
            config["img_transforms"] if config["img_transforms"] != "None" else None
        )
        self.point_cloud_transform = config["pc_transforms"]
        self.aug_rotate = config["pc_transforms"]
        self.aug_norm = config["pc_norm"]
        self.processed_dir = join(config["data_dir"], f'{self.config["resolution"]}m')
        self.data_dirs = {
            "train": join(
                config["data_dir"],
                f"{config['resolution']}m",
                "fusion_v2",
                "train",
                "superpixel",
            ),
            "val": join(
                config["data_dir"],
                f"{config['resolution']}m",
                "fusion_v2",
                "val",
                "superpixel",
            ),
            "test": join(
                config["data_dir"],
                f"{config['resolution']}m",
                "fusion_v2",
                "test",
                "superpixel",
            ),
        }

    def setup(self, stage=None):
        # Create datasets for train, validation, and test
        self.datasets = {}
        for split in ["train", "val", "test"]:
            data_dir = self.data_dirs[split]
            superpixel_files = [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith(".npz")
            ]
            self.datasets[split] = SuperpixelDataset(
                superpixel_files,
                rotate=None,
                normalization=self.aug_norm,
                image_transform=None,
                point_cloud_transform=None,
            )
            if split == "train":
                if not (
                    self.image_transform is None or self.point_cloud_transform is False
                ):
                    aug_img_dataset = SuperpixelDataset(
                        superpixel_files,
                        rotate=self.aug_rotate,
                        normalization=self.aug_norm,
                        image_transform=self.image_transform,
                        point_cloud_transform=False,
                    )
                    aug_pc_dataset = SuperpixelDataset(
                        superpixel_files,
                        rotate=self.aug_rotate,
                        normalization=self.aug_norm,
                        image_transform=None,
                        point_cloud_transform=self.point_cloud_transform,
                    )
                    self.datasets["train"] = torch.utils.data.ConcatDataset(
                        [self.datasets["train"], aug_img_dataset, aug_pc_dataset]
                    )

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["val"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        # Implement custom collate function if necessary
        batch = [b for b in batch if b is not None]  # Remove None samples if any

        images = torch.stack(
            [item["images"] for item in batch]
        )  # Shape: (batch_size, num_seasons, num_channels, 128, 128)
        point_clouds = torch.stack(
            [item["point_cloud"] for item in batch]
        )  # Shape: (batch_size, num_points, 3)
        pc_feats = torch.stack(
            [item["pc_feat"] for item in batch]
        )  # Shape: (batch_size, num_points, 3)
        labels = torch.stack(
            [item["label"] for item in batch]
        )  # Shape: (batch_size, num_classes)
        per_pixel_labels = torch.stack(
            [item["per_pixel_labels"] for item in batch]
        )  # Shape: (batch_size, num_classes, 128, 128)
        nodata_masks = torch.stack(
            [item["nodata_mask"] for item in batch]
        )  # Shape: (batch_size, 128, 128)

        return {
            "images": images,
            "point_cloud": point_clouds,
            "pc_feat": pc_feats,
            "label": labels,
            "per_pixel_labels": per_pixel_labels,
            "nodata_mask": nodata_masks,
        }
