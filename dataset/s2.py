import os
import rasterio
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np

class TreeSpeciesDataset(Dataset):
    def __init__(self, tile_names, processed_dir, datasets):
        """
        Args:
            tile_names (list): List of tile filenames to load.
            processed_dir (str): Base directory containing the processed data folders.
            datasets (list): List of dataset folder names to include (e.g., ['s2/spring', 's2/summer', ...]).
        """
        self.tile_names = tile_names
        self.processed_dir = processed_dir
        self.datasets = datasets  # List of dataset folder names
        
        # Calculate number of bands by inspecting the first tile of the first dataset
        example_file = os.path.join(self.processed_dir, datasets[0], tile_names[0])
        with rasterio.open(example_file) as src:
            self.n_bands = src.count

    def __len__(self):
        return len(self.tile_names)

    def __getitem__(self, idx):
        tile_name = self.tile_names[idx]
        input_data_list = []

        # Load data from each dataset (spring, summer, fall, winter, etc.)
        for dataset in self.datasets:
            dataset_path = os.path.join(self.processed_dir, dataset, tile_name)
            with rasterio.open(dataset_path) as src:
                input_data = src.read()  # Read the bands (num_bands, H, W)
                input_data_list.append(torch.from_numpy(input_data).float())  # Append each season's tensor to the list

        # Load the corresponding label (target species composition)
        label_path = os.path.join(self.processed_dir, 'labels/tiles_128', tile_name)
        
        with rasterio.open(label_path) as src:
            target_data = src.read()  # (num_bands, H, W)
            nodata_value_label = src.nodata  # NoData value for the labels

            # Create a NoData mask for the target data
            if nodata_value_label is not None:
                mask = np.any(target_data == nodata_value_label, axis=0)  # Collapse bands to (H, W)
            else:
                mask = np.zeros_like(target_data[0], dtype=bool)  # Assume all valid if no NoData value

        # Convert the target and mask to PyTorch tensors
        target_tensor = torch.from_numpy(target_data).float()  # Shape: (num_output_channels, H, W)
        mask_tensor = torch.from_numpy(mask).bool()  # Shape: (H, W)

        # Return the list of input tensors for each season, the target tensor, and the mask tensor
        return input_data_list, target_tensor, mask_tensor

class TreeSpeciesDataModule(pl.LightningDataModule):
    def __init__(self, tile_names, processed_dir, datasets_to_use, batch_size=4, num_workers=4):
        """
        Args:
            tile_names (dict): Dictionary with 'train', 'val', and 'test' keys containing lists of tile filenames to load.
            processed_dir (str): Directory where processed data is located.
            datasets_to_use (list): List of dataset names to include (e.g., ['s2/spring', 's2/summer', ...]).
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of workers for DataLoader.
        """
        super().__init__()
        self.tile_names = tile_names
        self.processed_dir = processed_dir
        self.datasets_to_use = datasets_to_use
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """
        Sets up the dataset for train, validation, and test splits.
        """
        # Create datasets for train, validation, and test
        self.train_dataset = TreeSpeciesDataset(self.tile_names['train'], self.processed_dir, self.datasets_to_use)
        self.val_dataset = TreeSpeciesDataset(self.tile_names['val'], self.processed_dir, self.datasets_to_use)
        self.test_dataset = TreeSpeciesDataset(self.tile_names['test'], self.processed_dir, self.datasets_to_use)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
