import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from pointnext import pointnext_s, PointNext, PointNextDecoder

class PointNeXtLightning(pl.LightningModule):
    def __init__(self, num_classes=9, learning_rate=1e-3):
        super(PointNeXtLightning, self).__init__()
        
        # Load PointNeXt backbone from torch-points3d
        encoder = pointnext_s(in_dim=3)
        self.pointnext = PointNext(num_classes, encoder=encoder, decoder=PointNextDecoder(encoder_dims=encoder.encoder_dims))
        
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, point_cloud):
        """
        Args:
            point_cloud: Input point cloud tensor (B, N, 3), where:
            B = Batch size, N = Number of points, 3 = (x, y, z) coordinates
        
        Returns:
            logits: Class logits for each point (B, N, num_classes)
        """
        return self.pointnext(point_cloud)
    
    def training_step(self, batch, batch_idx):
        point_clouds, targets, mask = batch
        outputs = self(point_clouds)  # Forward pass
        
        # Compute loss
        loss = self.criterion(outputs, targets.long())
        
        # Log training loss
        self.log('train_loss', loss)
        
        return loss

    def validation_step(self, batch, batch_idx):
        point_clouds, targets, mask = batch
        outputs = self(point_clouds)  # Forward pass
        
        # Compute loss
        loss = self.criterion(outputs, targets.long())
        
        # Log validation loss
        self.log('val_loss', loss)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # Example scheduler
        
        return [optimizer], [scheduler]