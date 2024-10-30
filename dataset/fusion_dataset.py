import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os
from data_utils.common import extract_polyids_from_mask, read_las
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split


class SuperpixelDataset(Dataset):
    def __init__(self, data_dir, point_cloud_dir, transform=None):
        self.file_paths = glob.glob(os.path.join(data_dir, '*.npz'))
        self.point_cloud_dir = point_cloud_dir
        self.transform = transform
        
        # Optional: Shuffle or sort file paths if needed
        self.file_paths.sort()
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load preprocessed data
        data = np.load(self.file_paths[idx])
        tile_image = data['tile_image']  # Shape: (bands, height, width)
        label_array = data['label_array']  # Shape: (classes, height, width)
        superpixel_mask = data['superpixel_mask']  # Shape: (height, width)
        nodata_mask = data['nodata_mask']  # Shape: (height, width)
        
        # Convert to tensors
        tile_image = torch.from_numpy(tile_image).float()
        label_array = torch.from_numpy(label_array).float()
        superpixel_mask = torch.from_numpy(superpixel_mask).long()
        nodata_mask = torch.from_numpy(nodata_mask)
        
        # Extract polyids from superpixel mask
        polyids = torch.unique(superpixel_mask)
        polyids = polyids[polyids != 0]  # Exclude background (0)
        
        # Load point cloud data for each polyid
        point_clouds = []
        for polyid in polyids:
            points = self.load_point_cloud(polyid.item())
            point_clouds.append(points)  # Shape: (num_points, num_features)
        
        # Stack point clouds into a tensor
        point_clouds = torch.stack(point_clouds)  # Shape: (num_superpixels, num_points, num_features)
        
        # Prepare the sample
        sample = {
            'tile_image': tile_image,  # (bands, H, W)
            'label_array': label_array,  # (classes, H, W)
            'superpixel_mask': superpixel_mask,  # (H, W)
            'nodata_mask': nodata_mask,  # (H, W)
            'point_clouds': point_clouds,  # (num_superpixels, num_points, num_features)
            'polyids': polyids  # (num_superpixels,)
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def load_point_cloud(self, polyid):
        filename = os.path.join(self.point_cloud_dir, f'{polyid}.npy')
        if not os.path.exists(filename):
            raise FileNotFoundError(f'Point cloud file for polyid {polyid} not found.')
        
        points = read_las(filename)
        points = torch.from_numpy(points).float()
        return points  # Tensor of shape (num_points, num_features)

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
            full_dataset, [train_size, val_size, test_size])
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
    
    def collate_fn(self, batch):
        # Implement custom collation to handle variable-size data
        batch_size = len(batch)
        
        # Collate tile images and labels (assuming same shape)
        tile_images = torch.stack([sample['tile_image'] for sample in batch])
        label_arrays = torch.stack([sample['label_array'] for sample in batch])
        superpixel_masks = torch.stack([sample['superpixel_mask'] for sample in batch])
        nodata_masks = torch.stack([sample['nodata_mask'] for sample in batch])
        
        # Collate point clouds and polyids
        point_clouds_list = [sample['point_clouds'] for sample in batch]
        polyids_list = [sample['polyids'] for sample in batch]
        
        # Since point clouds may have different numbers of superpixels, we handle them as lists
        batch = {
            'tile_image': tile_images,  # (batch_size, bands, H, W)
            'label_array': label_arrays,  # (batch_size, classes, H, W)
            'superpixel_mask': superpixel_masks,  # (batch_size, H, W)
            'nodata_mask': nodata_masks,  # (batch_size, H, W)
            'point_clouds': point_clouds_list,  # List of tensors
            'polyids': polyids_list  # List of tensors
        }
        
        return batch
    