import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from .pointNext import PointNextModel
from .loss import calc_loss

# from sklearn.metrics import r2_score

from torchmetrics.regression import R2Score


class PointNeXtLightning(pl.LightningModule):
    def __init__(self, params, in_dim):
        super(PointNeXtLightning, self).__init__()
        self.params = params
        if self.params["pc_norm"]:
            self.model = PointNextModel(self.params, in_dim=6)
        else:
            self.model = PointNextModel(self.params, in_dim=3)

        # Loss function and other parameters
        self.weights = self.params["train_weights"]  # Initialize on CPU
        self.train_r2 = R2Score()

        self.val_r2 = R2Score()

        self.test_r2 = R2Score()

        # Initialize metric storage for different stages (e.g., 'val', 'train')
        # self.val_r2 = []

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
        logits, _ = self.model(point_cloud, xyz)
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
        preds = F.softmax(logits, dim=1)

        # Compute the loss with the WeightedMSELoss, which will handle the weights
        if self.params["weighted_loss"] and stage == "train":
            self.weights = self.weights.to(logits.device)
            # Compute the loss with the WeightedMSELoss, which will handle the weights
            loss = calc_loss(targets, preds, self.weights)
        else:
            loss = F.mse_loss(preds, targets)

        # Calculate R² score for valid pixels
        # **Rounding Outputs for R² Score**
        # Round outputs to two decimal place/one
        # r2 = r2_score_torch(targets, torch.round(preds, decimals=1))
        preds = torch.round(preds, decimals=2)

        # Calculate R² and F1 score for valid pixels
        if stage == "train":
            r2 = self.train_r2(preds.view(-1), targets.view(-1))
        elif stage == "val":
            r2 = self.val_r2(preds.view(-1), targets.view(-1))
        else:
            r2 = self.test_r2(preds.view(-1), targets.view(-1))

        # Compute RMSE
        rmse = torch.sqrt(loss)

        # Store metrics dynamically based on stage (e.g., val_loss, val_r2)
        # if stage == "val":
        # getattr(self, f"{stage}_r2").append(r2)

        # Log the loss and R² score
        sync_state = True
        self.log(
            f"{stage}_loss",
            loss,
            logger=True,
            sync_dist=sync_state,
            on_step=True,
            on_epoch=(stage == "val"),
        )
        self.log(
            f"{stage}_r2",
            r2,
            logger=True,
            prog_bar=True,
            sync_dist=sync_state,
            on_step=True,
            on_epoch=(stage != "train"),
        )
        self.log(
            f"{stage}_rmse",
            rmse,
            logger=True,
            sync_dist=sync_state,
            on_step=True,
            on_epoch=(stage != "train"),
        )

        return loss

    def training_step(self, batch, batch_idx):
        point_cloud, xyz, targets = (
            batch  # Assuming batch contains (point_cloud, xyz, labels)
        )

        return self.foward_compute_loss_and_metrics(point_cloud, xyz, targets, "train")

    def validation_step(self, batch, batch_idx):
        point_cloud, xyz, targets = (
            batch  # Assuming batch contains (point_cloud, xyz, labels)
        )

        return self.foward_compute_loss_and_metrics(point_cloud, xyz, targets, "val")

    """
    def on_validation_epoch_end(self):
        # Compute the average of loss and r2 for the validation stage
        avg_r2 = torch.stack(self.val_r2).mean()
        
        # Log averaged metrics
        self.log("val_r2_epoch", avg_r2, prog_bar=True, sync_dist=True)
        
        # Clear the lists for the next epoch
        self.val_r2.clear()
    """

    def test_step(self, batch, batch_idx):
        point_cloud, xyz, targets = (
            batch  # Assuming batch contains (point_cloud, xyz, labels)
        )

        return self.foward_compute_loss_and_metrics(point_cloud, xyz, targets, "test")

    def configure_optimizers(self):
        if self.params["optimizer"] == "Adam":
            optimizer = Adam(
                self.parameters(),
                lr=self.params["learning_rate"],
                betas=(0.9, 0.999),
                eps=1e-08,
            )
        if self.params["optimizer"] == "AdamW":
            optimizer = AdamW(self.parameters(), lr=self.params["learning_rate"])
        else:
            optimizer = SGD(
                params=self.parameters(),
                lr=self.params["learning_rate"],
                momentum=self.params["momentum"],
                weight_decay=1e-4,
            )

        # Configure the scheduler based on the input parameter
        if self.params["scheduler"] == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer, patience=self.params["patience"], factor=0.5
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",  # Reduce learning rate when 'val_loss' plateaus
                },
            }
        elif self.params["scheduler"] == "steplr":
            scheduler = StepLR(optimizer, step_size=self.params["step_size"])
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.params["scheduler"] == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=50,
                eta_min=0,
                last_epoch=-1,
                verbose=False,
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer
