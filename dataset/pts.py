import os
import torch
from torch.utils.data import Subset, DataLoader, ConcatDataset
import pytorch_lightning as pl
from data_utils.common import PointCloudsInPickle
from data_utils.augment import AugmentPointCloudsInPickle

class PointCloudDataModule(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.trainset = None
        self.valset = None
        self.testset = None

    def prepare_data(self):
        # No need to download anything
        pass

    def setup(self, stage=None):
        # Load train dataset
        train_data_path = os.path.join(self.params["train_path"], str(self.params["num_points"]))
        train_pickle = self.params["train_pickle"]
        self.trainset = PointCloudsInPickle(train_data_path, train_pickle)

        trainset_idx = list(range(len(self.trainset)))
        rem = len(trainset_idx) % self.params["batch_size"]
        if rem <= 3:
            trainset_idx = trainset_idx[: len(trainset_idx) - rem]
            self.trainset = Subset(self.trainset, trainset_idx)
        
        # Apply augmentations if required
        if self.params["augment"]:
            aug_trainsets = [
                AugmentPointCloudsInPickle(train_data_path, train_pickle)
                for _ in range(self.params["n_augs"])
            ]
            self.trainset = ConcatDataset([self.trainset] + aug_trainsets)

        # Load validation dataset
        val_data_path = os.path.join(self.params["val_path"], str(self.params["num_points"]))
        val_pickle = self.params["val_pickle"]
        self.valset = PointCloudsInPickle(val_data_path, val_pickle)

        # Load test dataset (if eval mode is on)
        if self.params["eval"]:
            test_data_path = os.path.join(self.params["test_path"], str(self.params["num_points"]))
            test_pickle = self.params["test_pickle"]
            self.testset = PointCloudsInPickle(test_data_path, test_pickle)

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.params["batch_size"],
            shuffle=True,
            num_workers=self.params["num_workers"],
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.params["batch_size"],
            shuffle=False,
            num_workers=self.params["num_workers"],
            drop_last=True,
            pin_memory=True
        )

    def test_dataloader(self):
        if self.params["eval"]:
            return DataLoader(
                self.testset,
                batch_size=self.params["batch_size"],
                shuffle=False,
                num_workers=self.params["num_workers"],
                drop_last=True,
                pin_memory=True
            )
        return None