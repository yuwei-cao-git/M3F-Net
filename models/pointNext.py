import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from pointnext import pointnext_s, PointNext
from .loss import WeightedMSELoss

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
        self.criterion = WeightedMSELoss(self.weights)

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
        print(f"backbone output: {out.shape}")  # Check the output
        out = out.mean(dim=-1)
        out = self.act(out)
        logits = self.cls_head(out)
        return logits
    
    def training_step(self, batch, batch_idx):
        point_cloud, xyz, targets = batch  # Assuming batch contains (point_cloud, xyz, labels)
        point_cloud = point_cloud.permute(0, 2, 1)
        xyz = xyz.permute(0, 2, 1).float()
        logits = self.forward(point_cloud, xyz)
        
        # Move weights to the same device as logits
        self.weights = self.weights.to(logits.device)
        
        # Compute the loss with the WeightedMSELoss, which will handle the weights
        loss = self.criterion(F.softmax(logits, dim=1), targets)  # Pass weights directly
        
        # Log training loss
        self.log('train_loss', loss, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        point_cloud, xyz, targets = batch  # Assuming batch contains (point_cloud, xyz, labels)
        point_cloud = point_cloud.permute(0, 2, 1)
        xyz = xyz.permute(0, 2, 1).float()
        logits = self.forward(point_cloud, xyz)
        # Compute loss
        loss = F.mse_loss(F.softmax(logits, dim=1), targets)
        
        # Log validation loss
        self.log('val_loss', loss, sync_dist=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params["lr_c"])
        scheduler = StepLR(optimizer, step_size=self.params["step_size"], gamma=0.5)  # Example scheduler
        
        return [optimizer], [scheduler]