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

        # Apply point cloud transforms if any
        if self.point_cloud_transform:
            coords, xyz, label = self.point_cloud_transform(xyz, coords, label)

        # After applying transforms
        pc_feat = torch.from_numpy(coords).float()
        xyz = torch.from_numpy(xyz).float()
        label = torch.tensor(label, dtype=torch.float32)
        # Process and pad images
        processed_images = []
        processed_masks = []
        for img in superpixel_images:
            img = torch.from_numpy(
                img
            ).float()  # Convert image to tensor with shape [num_channels, height, width]

            # Create a nodata mask from the label array where the value indicates no-data (e.g., -1)
            nodata_mask = torch.from_numpy(
                label_img == -1
            ).bool()  # True where there's no data

            # Calculate padding to reach (128, 128)
            _, original_height, original_width = img.shape
            pad_height = self.target_size[0] - original_height
            pad_width = self.target_size[1] - original_width

            # Ensure padding is non-negative
            if pad_height < 0 or pad_width < 0:
                raise ValueError(
                    "Superpixel image is larger than the target size of 128x128."
                )

            # Pad symmetrically to reach the target size
            padding = (
                pad_width // 2,
                pad_width - pad_width // 2,
                pad_height // 2,
                pad_height - pad_height // 2,
            )

            img_padded = F.pad(img, padding, mode="constant", value=0)  # Pad the image
            mask_padded = F.pad(
                nodata_mask, padding, mode="constant", value=True
            )  # Pad the mask with True for NoData

            processed_images.append(img_padded)
            processed_masks.append(mask_padded)

        # Stack all seasonal images and masks along a new dimension
        images_tensor = torch.stack(
            processed_images, dim=0
        )  # Shape: [num_seasons, num_channels, 128, 128]
        masks_tensor = torch.stack(
            processed_masks, dim=0
        )  # Shape: [num_seasons, 128, 128]

        sample = {
            "images": images_tensor,  # Padded images of shape [num_seasons, num_channels, 128, 128]
            "nodata_mask": masks_tensor,  # Padded masks of shape [num_seasons, 128, 128]
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
        self.image_transform = config["img_transforms"]
        self.point_cloud_transform = config["pc_transforms"]

        self.processed_dir = config["data_dir"]
        self.data_dir = {
            "train": join(self.processed_dir, "train/superpixel"),
            "val": join(self.processed_dir, "val/superpixel"),
            "test": join(self.processed_dir, "test/superpixel"),
        }

    def setup(self, stage=None):
        # Create datasets for train, validation, and test
        self.train_dataset = SuperpixelDataset(
            self.data_dir["train"],
            image_transform=self.image_transform,
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
            collate_fn=custom_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            collate_fn=custom_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            collate_fn=custom_collate_fn,
        )


def custom_collate_fn(batch):
    batch = [b for b in batch if b is not None]  # Remove None samples

    # Extract data from batch
    images_list = [item["images"] for item in batch]  # List of lists of images
    point_clouds = [item["point_cloud"] for item in batch]
    pc_feats = [item["pc_feat"] for item in batch]
    labels = [item["label"] for item in batch]

    # Handle images
    images_batch = []
    masks_batch = []
    max_length = max(
        img.shape[-1] for images in images_list for img in images
    )  # Assuming shape [C, N]

    padded_images_batch = []
    for images in images_list:
        padded_images = []
        masks = []
        for img in images:
            length = img.shape[-1]
            pad_size = max_length - length
            if pad_size > 0:
                padding = torch.zeros((img.shape[0], pad_size), dtype=img.dtype)
                img_padded = torch.cat([img, padding], dim=-1)
            else:
                img_padded = img
            padded_images.append(img_padded)
            mask = torch.cat([torch.ones(length), torch.zeros(pad_size)])
            masks.append(mask)
        images_tensor = torch.stack(padded_images)  # Shape: [num_images, C, max_length]
        masks_tensor = torch.stack(masks)  # Shape: [num_images, max_length]
        images_batch.append(images_tensor)
        masks_batch.append(masks_tensor)
    images_batch = torch.stack(
        images_batch
    )  # Shape: [batch_size, num_images, C, max_length]
    masks_batch = torch.stack(masks_batch)

    # Handle point clouds (assuming fixed size)
    point_clouds = torch.stack(point_clouds)
    pc_feats = torch.stack(pc_feats)
    labels = torch.stack(labels)

    return {
        "images": images_batch,
        "image_masks": masks_batch,
        "point_cloud": point_clouds,
        "pc_feat": pc_feats,
        "label": labels,
    }
