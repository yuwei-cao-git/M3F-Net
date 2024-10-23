import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from pointnext import pointnext_s, PointNext
from .loss import calc_loss
# from torchmetrics import R2Score
from .metrics import r2_score_torch

class PointNeXtLightning(pl.LightningModule):
    def __init__(self, params, in_dim):
        super(PointNeXtLightning, self).__init__()
        self.params = params
        self.n_classes = len(params["classes"])
        
        # Initialize the PointNext encoder and decoder
        self.encoder = pointnext_s(in_dim=in_dim)  # Load the pointnext_s() as the encoder
        self.backbone = PointNext(self.params["emb_dims"], encoder=self.encoder)
        
        self.norm = nn.BatchNorm1d(self.params["emb_dims"])
        self.act = nn.ReLU()
        self.cls_head = nn.Sequential(
            nn.Linear(self.params["emb_dims"], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self.params["dropout"]),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(self.params["dropout"]),
            nn.Linear(256, self.n_classes),
        )
        
        # Loss function and other parameters
        self.weights = self.params["train_weights"]  # Initialize on CPU
        
        # Initialize metric storage for different stages (e.g., 'val', 'train')
        self.val_loss = []
        self.val_r2 = []

    def forward(self, point_cloud, xyz):
        """
        Args:
            point_cloud: Input point cloud tensor (B, N, 3), where:
            B = Batch size, N = Number of points, 3 = (x, y, z) coordinates
            xyz: The spatial coordinates of points
            category: Optional category tensor if categories are used
        Returns:
            logits: Class logits for each point (B, N, num_classes)
        """
        out = self.norm(self.backbone(point_cloud, xyz))
        out = out.mean(dim=-1)
        out = self.act(out)
        logits = self.cls_head(out)
        return logits
    
    def foward_compute_loss_and_metrics(self, point_cloud, xyz, targets, stage="val"):
        """
        Forward operations, computes the masked loss, R² score, and logs the metrics.

        Args:
        - stage: One of 'train', 'val', or 'test', used for logging purposes.

        Returns:
        - loss: The computed loss.
        """
        point_cloud = point_cloud.permute(0, 2, 1)
        xyz = xyz.permute(0, 2, 1).float()
        logits = self.forward(point_cloud, xyz)
        
        # Move weights to the same device as logits
        self.weights = self.weights.to(logits.device)
        
        # Compute the loss with the WeightedMSELoss, which will handle the weights
        if stage == "train":
            loss = calc_loss(targets, F.softmax(logits, dim=1), self.weights)
        else:
            loss = F.mse_loss(F.softmax(logits, dim=1), targets)
        
        # Calculate R² score for valid pixels
        # **Rounding Outputs for R² Score**
        # Round outputs to two decimal place
        valid_outputs = torch.round(F.softmax(logits, dim=1), decimals=2)
        r2 = r2_score_torch(targets, valid_outputs)
        
        # Compute RMSE
        rmse = torch.sqrt(loss)
        
        # Store metrics dynamically based on stage (e.g., val_loss, val_r2)
        if stage == "val":
            getattr(self, f"{stage}_loss").append(loss)
            getattr(self, f"{stage}_r2").append(r2)
        
        # Log the loss and R² score
        sync_state = True
        self.log(f'{stage}_loss', loss, logger=True, sync_dist=sync_state)
        self.log(f'{stage}_r2', r2, logger=True, prog_bar=True, sync_dist=sync_state)
        self.log(f'{stage}_rmse', rmse, logger=True, prog_bar=True, sync_dist=sync_state)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        point_cloud, xyz, targets = batch  # Assuming batch contains (point_cloud, xyz, labels)
        
        return self.foward_compute_loss_and_metrics(point_cloud, xyz, targets, "train")

    def validation_step(self, batch, batch_idx):
        point_cloud, xyz, targets = batch  # Assuming batch contains (point_cloud, xyz, labels)
        
        return self.foward_compute_loss_and_metrics(point_cloud, xyz, targets, "val")
    
    def on_validation_epoch_end(self):
        # Compute the average of loss and r2 for the validation stage
        avg_loss = torch.stack(self.val_loss).mean()
        avg_r2 = torch.stack(self.val_r2).mean()
        
        # Log averaged metrics
        self.log("val_loss_epoch", avg_loss, sync_dist=True)
        self.log("val_r2_epoch", avg_r2, sync_dist=True)
        
        # Clear the lists for the next epoch
        self.val_loss.clear()
        self.val_r2.clear()
    
    def test_step(self, batch, batch_idx):
        point_cloud, xyz, targets = batch  # Assuming batch contains (point_cloud, xyz, labels)
        
        return self.foward_compute_loss_and_metrics(point_cloud, xyz, targets, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params["lr_c"])
        scheduler = StepLR(optimizer, step_size=self.params["step_size"], gamma=0.5)  # Example scheduler
        if self.params['optimizer_c'] == "Adam":
            optimizer = Adam(self.parameters(),
                                        lr=self.params['lr_c'],
                                        betas=(0.9, 0.999), eps=1e-08)
        if self.params['optimizer_c'] == "AdamW":
            optimizer = AdamW(self.parameters(), lr=self.params['lr_c'])
        else:
            optimizer = SGD(params=self.parameters(),
                                        lr=self.params['lr_c'],
                                        momentum=self.params["momentum"],
                                        weight_decay=1e-4)

        
        # Configure the scheduler based on the input parameter
        if self.params["scheduler"] == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer, patience=self.params['patience'], factor=0.5
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',  # Reduce learning rate when 'val_loss' plateaus
                }
            }
        elif self.params["scheduler"] == "steplr":
            scheduler = StepLR(
                optimizer, step_size=self.params['step_size']
            )
            return{
                'optimizer': optimizer,
                'lr_scheduler': scheduler
            }
        elif self.params["scheduler"] == "cosine":
            scheduler = CosineAnnealingLR(
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