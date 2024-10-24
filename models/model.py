import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms.v2 as transforms

from .blocks import MF
# from .metrics import r2_score_torch
from torchmetrics import R2Score
from torcheval.metrics.functional import multiclass_f1_score

from .unet import UNet
from .ResUnet import ResUnet

# Updating UNet to incorporate residual connections and MF module
class Model(pl.LightningModule):
    def __init__(self, config):
        """
        Args:
            n_bands (int): Number of input channels (bands) for each season.
            n_classes (int): Number of output classes.
            use_mf (bool): Whether to use the MF module.
            use_residual (bool): Whether to use Residual connections in U-Net blocks.
            optimizer_type (str): Type of optimizer ('adam', 'sgd', etc.).
            learning_rate (float): Learning rate for the optimizer.
            scheduler_type (str): Type of scheduler ('plateau', etc.).
            scheduler_params (dict): Parameters for the scheduler (e.g., 'patience', 'factor' for ReduceLROnPlateau).
        """
        super(Model, self).__init__()
        self.config = config
        self.use_mf = self.config["use_mf"]
        self.use_residual = self.config["use_residual"]
        self.aug = self.config["transforms"]
        if self.config["resolution"] == 10:
            self.n_bands = 12
        else:
            self.n_bands = 9

        if self.use_mf:
            # MF Module for seasonal fusion (each season has `n_bands` channels)
            self.mf_module = MF(channels=self.n_bands)
            total_input_channels = 64  # MF module outputs 64 channels after processing four seasons
        else:
            total_input_channels = self.n_bands * 4  # If no MF module, concatenating all seasons directly

        # Define the U-Net architecture with or without Residual connections
        if self.use_residual:
            # Using ResUNet
            self.model = ResUnet(n_channels=total_input_channels, n_classes=self.config["n_classes"])
        else:
            # Using standard UNet
            self.model = UNet(n_channels=total_input_channels, n_classes=self.config["n_classes"])
        if self.aug == "random":
            self.transform = transforms.RandomApply(torch.nn.ModuleList([
                transforms.RandomCrop(size=(128,128)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                transforms.RandomRotation(degrees=(0,180)),
                transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))
            ]), p=0.3)
        else:
            self.transform = transforms.Compose([
                transforms.RandomCrop(size=(128,128)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                transforms.RandomRotation(degrees=(0,180)),
                transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))
            ])
        
        # Initialize metric storage for different stages (e.g., 'val', 'train')
        self.val_loss = []
        self.val_r2 = []
        
        # Loss function
        self.criterion = nn.MSELoss()
        self.r2_calc = R2Score()
        
        # Optimizer and scheduler settings
        self.optimizer_type = self.config["optimizer"]
        self.learning_rate = self.config["learning_rate"]
        self.scheduler_type = self.config["scheduler"]
        
        #self.save_hyperparameters(ignore=["loss_weight"])

    def forward(self, inputs):
        # Optionally pass inputs through MF module
        if self.aug is not None:
            inputs = self.transform(inputs)
        if self.use_mf:
            # Apply the MF module first to extract features from input
            fused_features = self.mf_module(inputs)
        else:
            # Concatenate all seasons directly if no MF module
            fused_features = torch.cat(inputs, dim=1)

        return self.model(fused_features)
    
    def apply_mask(self, outputs, targets, mask, multi_class=True):
        """
        Applies the mask to outputs and targets to exclude invalid data points.

        Args:
            outputs: Model predictions (batch_size, num_classes, H, W) for images or (batch_size, num_points, num_classes) for point clouds.
            targets: Ground truth labels (same shape as outputs).
            mask: Boolean mask indicating invalid data points (True for invalid).

        Returns:
            valid_outputs: Masked and reshaped outputs.
            valid_targets: Masked and reshaped targets.
        """
        # Expand the mask to match outputs and targets
        if multi_class:
            expanded_mask = mask.unsqueeze(1).expand_as(outputs)  # Shape: (batch_size, num_classes, H, W)
            num_classes = outputs.size(1)
        else:
            expanded_mask = mask

        # Apply mask to exclude invalid data points
        valid_outputs = outputs[~expanded_mask]
        valid_targets = targets[~expanded_mask]
        # Reshape to (-1, num_classes)
        if multi_class:
            valid_outputs = valid_outputs.view(-1, num_classes)
            valid_targets = valid_targets.view(-1, num_classes)

        return valid_outputs, valid_targets

    
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
        outputs = F.softmax(outputs, dim=1)
        
        valid_outputs, valid_targets = self.apply_mask(outputs, targets, masks, multi_class=True)
        
        # Compute the masked loss
        loss = self.criterion(valid_outputs, valid_targets)

        # Calculate R² score for valid pixels
        # **Rounding Outputs for R² Score**
        # Round outputs to two decimal place
        valid_outputs = torch.round(valid_outputs, decimals=1)
        # Renormalize after rounding to ensure outputs sum to 1 #TODO: validate
        # rounded_outputs = rounded_outputs / rounded_outputs.sum(dim=1, keepdim=True).clamp(min=1e-6)
        #r2 = r2_score_torch(valid_targets, valid_outputs)
        r2 = self.r2_calc(valid_outputs.view(-1), valid_targets.view(-1))
        
        # Compute RMSE
        rmse = torch.sqrt(loss)
        
        # F1 Score Calculation
        # Convert outputs and targets to leading class labels by taking argmax
        pred_labels = torch.argmax(outputs, dim=1)
        true_labels = torch.argmax(targets, dim=1)
        # Apply mask
        valid_preds, valid_true = self.apply_mask(pred_labels, true_labels, masks, multi_class=False)
        f1 = multiclass_f1_score(valid_preds, valid_true, num_classes=self.config["n_classes"])
        
        # Store metrics dynamically based on stage (e.g., val_loss, val_r2, val_rmse)
        if stage == "val":
            getattr(self, f"{stage}_r2").append(r2)
        
        # Log the loss and R² score
        sync_state = True
        self.log(f'{stage}_loss', loss, logger=True, sync_dist=sync_state)
        self.log(f'{stage}_r2', r2, logger=True, prog_bar=True, sync_dist=sync_state)
        self.log(f'{stage}_rmse', rmse, logger=True, sync_dist=sync_state)
        self.log(f'{stage}_f1', f1, logger=True, sync_dist=sync_state)

        return loss
    
    def training_step(self, batch, batch_idx):
        inputs, targets, masks = batch
        outputs = self(inputs)  # Forward pass #[batch_size, n_classes, height, width]
        
        return self.compute_loss_and_metrics(outputs, targets, masks, stage="train")

    def validation_step(self, batch, batch_idx):
        inputs, targets, masks = batch
        outputs = self(inputs)  # Forward pass
        return self.compute_loss_and_metrics(outputs, targets, masks, stage="val")
    
    def on_validation_epoch_end(self):
        # Compute the average of r2 for the validation stage
        avg_r2 = torch.stack(self.val_r2).mean()
        sys_r2 = self.r2_calc.compute()
        
        # Log averaged metrics
        self.log("val_r2_epoch", avg_r2, sync_dist=True)
        self.log("sys_r2", sys_r2, sync_dist=True)
        
        # Clear the lists for the next epoch
        self.val_r2.clear()
        self.r2_calc.reset()
    
    def test_step(self, batch, batch_idx):
        inputs, targets, masks = batch
        outputs = self(inputs)  # Forward pass

        return self.compute_loss_and_metrics(outputs, targets, masks, stage="test")
    
    def configure_optimizers(self):
        # Choose the optimizer based on input parameter
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, 
                                        betas=(0.9, 0.999), eps=1e-08)
        elif self.optimizer_type == "adamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, 
                                        weight_decay=1e-4)
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

        # Configure the scheduler based on the input parameter
        if self.scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=10, factor=0.5
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',  # Reduce learning rate when 'val_loss' plateaus
                }
            }
        elif self.scheduler_type == "asha":
            return optimizer
        elif self.scheduler_type == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=20
            )
            return{
                'optimizer': optimizer,
                'lr_scheduler': scheduler
            }
        elif self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=50,
                eta_min=0,
                last_epoch=-1,
                verbose=False,
            )
            return{
                'optimizer': optimizer,
                'lr_scheduler': scheduler
            }
        else:
            return optimizer
