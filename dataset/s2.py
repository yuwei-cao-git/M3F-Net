import os
import torch
import rasterio
from torch.utils.data import Dataset
import numpy as np

class TreeSpeciesDataset(Dataset):
    def __init__(self, tiles_dir, labels_dir, tile_names):
        """
        Args:
            tiles_dir (str): Directory containing the input tiles (12 bands).
            labels_dir (str): Directory containing the label tiles (9 bands).
            tile_names (list): List of tile filenames to load (e.g., 'tile_x_y.tif').
        """
        self.tiles_dir = tiles_dir
        self.labels_dir = labels_dir
        self.tile_names = tile_names

    def __len__(self):
        return len(self.tile_names)

    def __getitem__(self, idx):
        tile_name = self.tile_names[idx]

        # Load the input tile (12 bands)
        input_tile_path = os.path.join(self.tiles_dir, tile_name)
        with rasterio.open(input_tile_path) as src:
            input_data = src.read()  # Shape: (12, H, W)
            nodata_value_input = src.nodata  # NoData value for the input

        # Load the target tile (9 bands)
        label_tile_path = os.path.join(self.labels_dir, tile_name)
        with rasterio.open(label_tile_path) as src:
            target_data = src.read()  # Shape: (9, H, W)
            nodata_value_target = src.nodata  # NoData value for the target

        # Create masks for NoData pixels (True where NoData)
        input_mask = input_data == nodata_value_input
        target_mask = target_data == nodata_value_target

        # Convert NoData to -1 in both input and target for easier handling
        input_data = np.where(input_mask,-1, input_data)
        target_data = np.where(target_mask, -1, target_data)

        # Create a combined mask (where either input or target has NoData)
        combined_mask = np.any(input_mask, axis=0) | np.any(target_mask, axis=0)

        # Convert data to PyTorch tensors
        input_tensor = torch.from_numpy(input_data).float()
        target_tensor = torch.from_numpy(target_data).float()
        mask_tensor = torch.from_numpy(combined_mask).bool()  # Convert mask to PyTorch boolean tensor

        return input_tensor, target_tensor, mask_tensor