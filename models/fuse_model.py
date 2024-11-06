import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .blocks import MF
from .unet import UNet
from .ResUnet import ResUnet
from .pointNext import PointNextModel

from torchmetrics.regression import R2Score
from torchmetrics.classification import MulticlassF1Score
from .loss import MaskedMSELoss, apply_mask


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
            out_channels = 1024
        else:
            # Using standard UNet
            self.s2_model = UNet(
                n_channels=total_input_channels, n_classes=self.config["n_classes"]
            )
            out_channels = 512

        # Initialize point cloud stream model
        self.pointnext = PointNextModel(self.config, in_dim=3)

        # Fusion and classification layers with additional linear layer
        self.fc1 = nn.Linear(out_channels + self.config["emb_dims"], 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(128, self.config["n_classes"])

        # Define loss functions
        self.criterion = nn.MSELoss()
        self.img_criterion = MaskedMSELoss()

        # Metrics
        self.train_r2 = R2Score()
        self.train_f1 = MulticlassF1Score(num_classes=self.config["n_classes"])

        self.val_r2 = R2Score()
        self.val_f1 = MulticlassF1Score(num_classes=self.config["n_classes"])

        self.test_r2 = R2Score()
        self.test_f1 = MulticlassF1Score(num_classes=self.config["n_classes"])

        # Optimizer and scheduler settings
        self.optimizer_type = self.config["optimizer"]
        self.scheduler_type = self.config["scheduler"]

    def forward(self, images, pc_feat, xyz):
        # Forward pass through UNet
        if self.use_mf:
            # Apply the MF module first to extract features from input
            fused_features = self.mf_module(images)
        else:
            # Concatenate all seasons directly if no MF module
            batch_size, num_seasons, num_channels, width, height = images.shape

            # Reshape images to merge `num_seasons` and `num_channels` by concatenating along channels
            fused_features = images.view(
                batch_size, num_seasons * num_channels, width, height
            )  # torch.Size([4, 36, 128, 128])
        image_outputs, img_emb = self.s2_model(
            fused_features
        )  # torch.Size([4, 9, 128, 128])
        # Forward pass through PointNet
        point_outputs, pc_emb = self.pointnext(pc_feat, xyz)  # torch.Size([4, 9])
        # Flatten image features for fusion
        image_features_pooled = F.adaptive_avg_pool2d(img_emb, (1, 1)).view(
            batch_size, -1
        )  # Shape: (batch_size, feature_dim)

        # Pool over points
        point_cloud_features = torch.max(pc_emb, dim=2)[
            0
        ]  # Shape: (batch_size, feature_dim)
        # Concatenate features
        combined_features = torch.cat(
            (image_features_pooled, point_cloud_features), dim=1
        )

        # Fusion and classification with additional layers and regularization
        x = F.relu(self.bn1(self.fc1(combined_features)))  # [batch_size, 512]
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))  # [batch_size, 128]
        x = self.dropout2(x)
        class_output = self.fc3(x)  # [batch_size, num_classes]

        return image_outputs, point_outputs, class_output

    def foward_and_metrics(
        self, images, img_masks, pc_feat, point_clouds, labels, pixel_labels, stage
    ):
        """
        Forward operations, computes the masked loss, R² score, and logs the metrics.

        Args:
        - stage: One of 'train', 'val', or 'test', used for logging purposes.

        Returns:
        - loss: The computed loss.
        """
        pc_feat = pc_feat.permute(0, 2, 1)
        point_clouds = point_clouds.permute(0, 2, 1)
        pixel_logits, pc_logits, fuse_logits = self.forward(
            images, pc_feat, point_clouds
        )
        pc_preds = F.softmax(pc_logits, dim=1)
        pixel_preds = F.softmax(pixel_logits, dim=1)
        fuse_preds = F.softmax(fuse_logits, dim=1)
        # Convert preds and labels to leading class labels by taking argmax
        pred_pc_labels = torch.argmax(pc_preds, dim=1)
        pred_lead_pixel_labels = torch.argmax(pixel_preds, dim=1)
        true_labels = torch.argmax(labels, dim=1)
        true_lead_pixel_labels = torch.argmax(pixel_labels, dim=1)

        valid_pixel_preds, valid_labels = apply_mask(
            pixel_preds, pixel_labels, img_masks
        )
        valid_pixel_labels, valid_true_labels = apply_mask(
            pred_lead_pixel_labels, true_lead_pixel_labels, img_masks, multi_class=False
        )

        # Compute loss
        loss_pixel = self.criterion(valid_pixel_preds, valid_labels)
        loss_point = self.criterion(pc_preds, labels)
        loss_image = self.criterion(fuse_preds, labels)
        loss = loss_image + loss_point + loss_pixel  # Adjust weights as needed

        # Compute RMSE
        rmse = torch.sqrt(loss)

        # Compute R² score & f1 score of leading species
        pc_preds = torch.round(pc_preds, decimals=1)
        fuse_preds = torch.round(fuse_preds, decimals=1)
        valid_pixel_preds = torch.round(valid_pixel_preds, decimals=1)
        if stage == "train":
            pc_r2 = self.train_r2(pc_preds.view(-1), labels.view(-1))
            pixel_r2 = self.train_r2(valid_pixel_preds.view(-1), valid_labels.view(-1))
            fuse_r2 = self.train_r2(fuse_preds.view(-1), labels.view(-1))
            pc_f1 = self.train_f1(pred_pc_labels, true_labels)
            img_f1 = self.train_f1(valid_pixel_labels, valid_true_labels)
        elif stage == "val":
            pc_r2 = self.val_r2(pc_preds.view(-1), labels.view(-1))
            pixel_r2 = self.val_r2(valid_pixel_preds.view(-1), valid_labels.view(-1))
            fuse_r2 = self.val_r2(fuse_preds.view(-1), labels.view(-1))
            pc_f1 = self.val_f1(pred_pc_labels, true_labels)
            img_f1 = self.val_f1(valid_pixel_labels, valid_true_labels)
        else:
            pc_r2 = self.test_r2(pc_preds.view(-1), labels.view(-1))
            pixel_r2 = self.test_r2(valid_pixel_preds.view(-1), valid_labels.view(-1))
            fuse_r2 = self.test_r2(fuse_preds.view(-1), labels.view(-1))
            pc_f1 = self.test_f1(pred_pc_labels, true_labels)
            img_f1 = self.test_f1(valid_pixel_labels, valid_true_labels)

        # Log and return loss
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
            f"{stage}_rmse",
            rmse,
            logger=True,
            sync_dist=sync_state,
            on_step=True,
            on_epoch=(stage != "train"),
        )
        self.log(
            f"pc_{stage}_r2",
            pc_r2,
            logger=True,
            prog_bar=True,
            sync_dist=sync_state,
            on_step=True,
            on_epoch=(stage != "train"),
        )
        self.log(
            f"fuse_{stage}_r2",
            fuse_r2,
            logger=True,
            prog_bar=True,
            sync_dist=sync_state,
            on_step=True,
            on_epoch=(stage != "train"),
        )
        self.log(
            f"img_{stage}_r2",
            pixel_r2,
            logger=True,
            prog_bar=True,
            sync_dist=sync_state,
            on_step=True,
            on_epoch=(stage != "train"),
        )
        self.log(
            f"pc_{stage}_f1",
            pc_f1,
            logger=True,
            prog_bar=True,
            sync_dist=sync_state,
            on_step=True,
            on_epoch=(stage != "train"),
        )
        self.log(
            f"img_{stage}_f1",
            img_f1,
            logger=True,
            prog_bar=True,
            sync_dist=sync_state,
            on_step=True,
            on_epoch=(stage != "train"),
        )

        return loss

    def training_step(self, batch, batch_idx):
        images = batch[
            "images"
        ]  # Shape: (batch_size, num_seasons, num_channels, 128, 128)
        point_clouds = batch["point_cloud"]  # Shape: (batch_size, num_points, 3)
        labels = batch["label"]  # Shape: (batch_size, num_classes)
        per_pixel_labels = batch[
            "per_pixel_labels"
        ]  # Shape: (batch_size, num_classes, 128, 128)
        image_masks = batch["nodata_mask"]
        pc_feat = batch["pc_feat"]  # Shape: [batch_size, num_points, 3]

        loss = self.foward_and_metrics(
            images,
            image_masks,
            pc_feat,
            point_clouds,
            labels,
            per_pixel_labels,
            stage="train",
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch[
            "images"
        ]  # Shape: (batch_size, num_seasons, num_channels, 128, 128)
        point_clouds = batch["point_cloud"]  # Shape: (batch_size, num_points, 3)
        labels = batch["label"]  # Shape: (batch_size, num_classes)
        per_pixel_labels = batch[
            "per_pixel_labels"
        ]  # Shape: (batch_size, num_classes, 128, 128)
        image_masks = batch["nodata_mask"]
        pc_feat = batch["pc_feat"]  # Shape: [batch_size, num_points, 3]

        loss = self.foward_and_metrics(
            images,
            image_masks,
            pc_feat,
            point_clouds,
            labels,
            per_pixel_labels,
            stage="val",
        )
        return loss

    def test_step(self, batch, batch_idx):
        images = batch[
            "images"
        ]  # Shape: (batch_size, num_seasons, num_channels, 128, 128)
        point_clouds = batch["point_cloud"]  # Shape: (batch_size, num_points, 3)
        labels = batch["label"]  # Shape: (batch_size, num_classes)
        per_pixel_labels = batch[
            "per_pixel_labels"
        ]  # Shape: (batch_size, num_classes, 128, 128)
        image_masks = batch["nodata_mask"]
        pc_feat = batch["pc_feat"]  # Shape: [batch_size, num_points, 3]

        loss = self.foward_and_metrics(
            images,
            image_masks,
            pc_feat,
            point_clouds,
            labels,
            per_pixel_labels,
            stage="test",
        )
        return loss

    def configure_optimizers(self):
        # Choose the optimizer based on input parameter
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.config["learning_rate"],
                betas=(0.9, 0.999),
                eps=1e-08,
            )
        elif self.optimizer_type == "adamW":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.config["learning_rate"]
            )
        elif self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.config["learning_rate"],
                momentum=self.config["momentum"],
                weight_decay=self.config["weight_decay"],
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

        # Configure the scheduler based on the input parameter
        if self.scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=self.config["patience"], factor=0.5
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",  # Reduce learning rate when 'val_loss' plateaus
                },
            }
        elif self.scheduler_type == "asha":
            return optimizer
        elif self.scheduler_type == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.config["step_size"]
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=50,
                eta_min=0,
                last_epoch=-1,
                verbose=False,
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer