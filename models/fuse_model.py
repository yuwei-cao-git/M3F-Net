import torch
import torch.nn as nn
import pytorch_lightning as pl

from .blocks import MF
from pointnext import pointnext_s, PointNext, pointnext_b, pointnext_l, pointnext_xl
from .unet import UNet
from .ResUnet import ResUnet
from .pointNext import PointNext

from torchmetrics.regression import R2Score
from torchmetrics.classification import MulticlassF1Score


class SuperpixelModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.use_mf = self.config["use_mf"]
        self.use_residual = self.config["use_residual"]
        # Initialize s2 model
        if self.config["resolution"] == 10:
            self.n_bands = 12
        else:
            self.n_bands = 9

        if self.use_mf:
            # MF Module for seasonal fusion (each season has `n_bands` channels)
            self.mf_module = MF(channels=self.n_bands)
            total_input_channels = (
                64  # MF module outputs 64 channels after processing four seasons
            )
        else:
            total_input_channels = (
                self.n_bands * 4
            )  # If no MF module, concatenating all seasons directly

        # Define the U-Net architecture with or without Residual connections
        if self.use_residual:
            # Using ResUNet
            self.s2_model = ResUnet(
                n_channels=total_input_channels, n_classes=self.config["n_classes"]
            )
        else:
            # Using standard UNet
            self.s2_model = UNet(
                n_channels=total_input_channels, n_classes=self.config["n_classes"]
            )

        # Initialize point cloud stream model
        self.pointnext = PointNext(self.config, in_dim=3)

        # Define loss functions
        self.criterion = nn.MSELoss()

        # Metrics
        self.train_r2 = R2Score()
        self.train_f1 = MulticlassF1Score(num_classes=self.config["n_classes"])

        self.val_r2 = R2Score()
        self.val_f1 = MulticlassF1Score(num_classes=self.config["n_classes"])

        self.test_r2 = R2Score()
        self.test_f1 = MulticlassF1Score(num_classes=self.config["n_classes"])

    def forward(self, images, pc_feat, xyz):
        # Forward pass through UNet
        image_outputs = self.s2_model(images)
        # Forward pass through PointNet
        point_outputs = self.pointnext(pc_feat, xyz)
        return image_outputs, point_outputs

    def training_step(self, batch, batch_idx):
        images = batch["images"]
        pc_feat = batch["pc_feat"]
        point_clouds = batch["point_cloud"]
        labels = batch["label"]

        # Forward pass
        image_outputs, point_outputs = self(images, pc_feat, point_clouds)

        # Compute loss
        loss_image = self.criterion(image_outputs, labels)
        loss_point = self.criterion(point_outputs, labels)
        loss = loss_image + loss_point  # Adjust weights as needed

        # Log and return loss
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])
        return optimizer
