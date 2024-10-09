import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from blocks import MF, ResidualBlock
from loss import MaskedMSELoss
from metrics import r2_score_torch

# Updating UNet to incorporate residual connections and MF module
class ResUNet_MF(pl.LightningModule):
    def __init__(self, n_bands=13, out_channels=9, use_mf=False, use_residual=False, optimizer_type="adam", learning_rate=1e-3, scheduler_type=None, scheduler_params=None):
        """
        Args:
            n_bands (int): Number of input channels (bands) for each season.
            out_channels (int): Number of output channels.
            use_mf (bool): Whether to use the MF module.
            use_residual (bool): Whether to use Residual connections in U-Net blocks.
            optimizer_type (str): Type of optimizer ('adam', 'sgd', etc.).
            learning_rate (float): Learning rate for the optimizer.
            scheduler_type (str): Type of scheduler ('plateau', etc.).
            scheduler_params (dict): Parameters for the scheduler (e.g., 'patience', 'factor' for ReduceLROnPlateau).
        """
        super(ResUNet_MF, self).__init__()

        self.use_mf = use_mf
        self.use_residual = use_residual

        if self.use_mf:
            # MF Module for seasonal fusion (each season has `n_bands` channels)
            self.mf_module = MF(channels=n_bands)
            total_input_channels = 64  # MF module outputs 64 channels after processing four seasons
        else:
            total_input_channels = n_bands * 4  # If no MF module, concatenating all seasons directly

        # Define the U-Net architecture with or without Residual connections
        if self.use_residual:
            self.enc_conv0 = ResidualBlock(total_input_channels, 64)
            self.enc_conv1 = ResidualBlock(64, 128)
            self.enc_conv2 = ResidualBlock(128, 256)
            self.enc_conv3 = ResidualBlock(256, 512)
            self.dec_conv3 = ResidualBlock(512, 256)
            self.dec_conv2 = ResidualBlock(256, 128)
            self.dec_conv1 = ResidualBlock(128, 64)
        else:
            self.enc_conv0 = nn.Conv2d(total_input_channels, 64, kernel_size=3, padding=1)
            self.enc_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.enc_conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.enc_conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
            self.dec_conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
            self.dec_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
            self.dec_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.dec_conv0 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)  # Output layer
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Loss and learning rate
        self.learning_rate = learning_rate
        self.criterion = MaskedMSELoss()

        # Optimizer and scheduler types
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.scheduler_params = scheduler_params if scheduler_params else {}
        
        self.save_hyperparameters()

    def forward(self, inputs):
        # Optionally pass inputs through MF module
        if self.use_mf:
            # Apply the MF module first to extract features from input
            spring, summer, fall, winter = inputs  # Unpack the individual datasets
            # Process through the MF module
            fused_features = self.mf_module([spring, summer, fall, winter])
        else:
            # Concatenate all seasons directly if no MF module
            fused_features = torch.cat(inputs, dim=1)

        # U-Net forward pass (with or without residual connections)
        x1 = F.relu(self.enc_conv0(fused_features))
        x2 = self.pool(x1)
        x2 = F.relu(self.enc_conv1(x2))
        x3 = self.pool(x2)
        x3 = F.relu(self.enc_conv2(x3))
        x4 = self.pool(x3)
        x4 = F.relu(self.enc_conv3(x4))

        x = self.up(x4)
        x = F.relu(self.dec_conv3(x))
        x = self.up(x)
        x = F.relu(self.dec_conv2(x))
        x = self.up(x)
        x = F.relu(self.dec_conv1(x))
        x = self.dec_conv0(x)  # Output layer (no activation here)

        return x
    
    def compute_loss_and_metrics(self, outputs, targets, masks, stage="val"):
        """
        Computes the masked loss, R² score, and logs the metrics.

        Args:
        - outputs: Predicted values (batch_size, num_channels, H, W)
        - targets: Ground truth values (batch_size, num_channels, H, W)
        - masks: Boolean mask indicating NoData pixels (batch_size, H, W)
        - stage: One of 'train', 'val', or 'test', used for logging purposes.

        Returns:
        - loss: The computed masked loss.
        """
        
        # Expand the mask to match the number of channels in outputs and targets
        expanded_mask = masks.unsqueeze(1).expand_as(outputs)  # Shape: (batch_size, num_channels, H, W)

        # Exclude NoData pixels by applying the mask (keep only valid pixels)
        valid_outputs = outputs.masked_select(~expanded_mask).view(-1, outputs.size(1))
        valid_targets = targets.masked_select(~expanded_mask).view(-1, targets.size(1))

        # Compute the masked loss
        loss = self.criterion(outputs, targets, masks)

        # Calculate R² score for valid pixels
        r2 = r2_score_torch(valid_targets, valid_outputs)

        # Log the loss and R² score
        self.log(f'{stage}_loss', loss, logger=True)
        self.log(f'{stage}_r2', r2, logger=True, prog_bar=True)

        return loss
    
    def training_step(self, batch, batch_idx):
        inputs, targets, masks = batch
        outputs = self(inputs)  # Forward pass
        
        return self.compute_loss_and_metrics(outputs, targets, masks, stage="train")

    def validation_step(self, batch, batch_idx):
        inputs, targets, masks = batch
        outputs = self(inputs)  # Forward pass
        
        return self.compute_loss_and_metrics(outputs, targets, masks, stage="val")
    
    def test_step(self, batch, batch_idx):
        inputs, targets, masks = batch
        outputs = self(inputs)  # Forward pass

        return self.compute_loss_and_metrics(outputs, targets, masks, stage="test")
    
    def configure_optimizers(self):
        # Choose the optimizer based on input parameter
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

        # Configure the scheduler based on the input parameter
        if self.scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **self.scheduler_params
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',  # Reduce learning rate when 'val_loss' plateaus
                }
            }
        else:
            return optimizer