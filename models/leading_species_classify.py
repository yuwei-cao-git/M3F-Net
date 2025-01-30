import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from .blocks import MambaFusionBlock, MLPBlock
from .unet import UNet
from .pointNext import PointNextModel

from torchmetrics.classification import (
    MulticlassF1Score,
    ConfusionMatrix,
    MulticlassAccuracy,
)
from .loss import focal_loss_multiclass, apply_mask

import os
import pandas as pd
import numpy as np


class SuperpixelModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.f = self.config["linear_layers_dims"]  # f = [512, 128]
        # Initialize s2 model
        if self.config["resolution"] == 10:
            self.n_bands = 12
        else:
            self.n_bands = 9

        total_input_channels = (
            self.n_bands * 4
        )  # If no MF module, concatenating all seasons directly

        # Using standard UNet
        self.s2_model = UNet(
            n_channels=total_input_channels,
            n_classes=self.config["n_classes"],
            return_logits=True,
        )

        # Initialize point cloud stream model
        self.pc_model = PointNextModel(
            self.config, in_dim=6 if self.config.get("pc_norm", False) else 3
        )
        if self.config["mamba_fuse"]:
            self.fuse_head = MambaFusionBlock(
                in_img_chs=512,
                in_pc_chs=(self.config["emb_dims"]),
                dim=self.config["fusion_dim"],
                hidden_ch=self.config["linear_layers_dims"],
                num_classes=self.config["n_classes"],
                drop=self.config["dp_fuse"],
                return_logits=True,
            )
        else:
            # Fusion and classification layers with additional linear layer
            in_ch = 512 + self.config["emb_dims"]
            self.fuse_head = MLPBlock(
                config, in_ch, self.config["linear_layers_dims"], return_logits=True
            )
        # Define loss functions
        if self.config["loss"] == "wce":
            # Loss function and other parameters
            self.class_weights = torch.tensor(
                [
                    0.64776265,
                    0.00496458,
                    0.01286589,
                    0.01814702,
                    0.00474632,
                    0.00590281,
                    0.04318462,
                    0.00104648,
                    0.26137963,
                ]
            )

        # Metrics
        self.train_f1 = MulticlassF1Score(
            num_classes=self.config["n_classes"], average="weighted"
        )
        self.train_oa = MulticlassAccuracy(num_classes=self.config["n_classes"])

        self.val_f1 = MulticlassF1Score(
            num_classes=self.config["n_classes"], average="weighted"
        )
        self.val_oa = MulticlassAccuracy(
            num_classes=self.config["n_classes"], average="micro"
        )

        self.test_f1 = MulticlassF1Score(
            num_classes=self.config["n_classes"], average="weighted"
        )
        self.test_oa = MulticlassAccuracy(
            num_classes=self.config["n_classes"], average="micro"
        )

        self.confmat = ConfusionMatrix(
            task="multiclass", num_classes=self.config["n_classes"]
        )

        # Optimizer and scheduler settings
        self.optimizer_type = self.config["optimizer"]
        self.scheduler_type = self.config["scheduler"]
        self.lr = self.config["lr"]

        self.best_test_f1 = 0.0
        self.best_test_outputs = None
        self.validation_step_outputs = []

    def forward(self, images, pc_feat, xyz):
        image_logits = None
        img_emb = None
        pc_logits = None
        pc_emb = None

        batch_size, num_seasons, num_channels, width, height = images.shape
        fused_features = images.view(
            batch_size, num_seasons * num_channels, width, height
        )
        image_logits, img_emb = self.s2_model(fused_features)

        # Process point clouds
        pc_logits, pc_emb = self.pc_model(pc_feat, xyz)
        pc_emb = torch.max(pc_emb, dim=2)[0]  # Shape: (batch_size, feature_dim)
        # Fusion and classification
        fuse_logits = self.fuse_head(img_emb, pc_emb)
        return image_logits, pc_logits, fuse_logits

    def forward_and_metrics(
        self, images, img_masks, pc_feat, point_clouds, labels, pixel_labels, stage
    ):
        """
        Forward operations, computes the masked loss, R² score, and logs the metrics.

        Args:
        - images: Image data
        - img_masks: Masks for images
        - pc_feat: Point cloud features
        - point_clouds: Point cloud coordinates
        - labels: Ground truth labels for classification
        - pixel_labels: Ground truth labels for per-pixel predictions
        - stage: One of 'train', 'val', or 'test', used to select appropriate metrics and logging

        Returns:
        - loss: The computed loss
        """
        # Permute point cloud data if available
        pc_feat = pc_feat.permute(0, 2, 1) if pc_feat is not None else None
        point_clouds = (
            point_clouds.permute(0, 2, 1) if point_clouds is not None else None
        )

        # Forward pass
        pixel_logits, pc_logits, fuse_logits = self.forward(
            images, pc_feat, point_clouds
        )

        true_labels = torch.argmax(labels, dim=1)
        loss = 0
        logs = {}

        # Select appropriate metric instances based on the stage
        if stage == "train":
            f1_metric = self.train_f1
            oa_metric = self.train_oa
        elif stage == "val":
            f1_metric = self.val_f1
            oa_metric = self.val_oa
        else:  # stage == "test"
            f1_metric = self.test_f1
            oa_metric = self.test_oa

        # Compute point cloud loss
        if self.config["loss"] == "wce" and stage == "train":
            loss_point = F.nll_loss(
                pc_logits, true_labels, weight=self.class_weights.to(pc_logits.device)
            )
        elif self.config["loss"] == "ae" and stage == "train":
            pred_lead_pc_labels = torch.argmax(pc_logits, dim=1)
            correct = (pred_lead_pc_labels.view(-1) == true_labels.view(-1)).float()
            loss_point = 1 - correct.mean()  # 1 - accuracy as pseudo-loss
        elif self.config["loss"] == "focal" and stage == "train":
            loss_point = focal_loss_multiclass(pc_logits, true_labels)
        else:
            loss_point = F.nll_loss(pc_logits, true_labels)
        loss += loss_point * 0.5

        # Log metrics
        logs.update({f"pc_{stage}_loss": loss_point})

        # Image stream
        pred_lead_pixel_labels = torch.argmax(pixel_logits, dim=1)
        true_lead_pixel_labels = torch.argmax(pixel_labels, dim=1)
        _, valid_pixel_lead_true = apply_mask(
            pred_lead_pixel_labels,
            true_lead_pixel_labels,
            img_masks,
            multi_class=False,
            keep_shp=True,
        )
        if self.config["loss"] == "wce" and stage == "train":
            loss_pixel_leads = F.nll_loss(
                pixel_logits,
                valid_pixel_lead_true,
                weight=self.class_weights.to(pc_logits.device),
                ignore_index=255,
            )
        elif self.config["loss"] == "ae" and stage == "train":
            valid_pixel_lead_preds, valid_pixel_lead_true = apply_mask(
                pred_lead_pixel_labels,
                true_lead_pixel_labels,
                img_masks,
                multi_class=False,
                keep_shp=False,
            )
            correct = (
                valid_pixel_lead_preds.view(-1) == valid_pixel_lead_true.view(-1)
            ).float()
            loss_pixel_leads = 1 - correct.mean()  # 1 - accuracy as pseudo-loss
        elif self.config["loss"] == "focal" and stage == "train":
            loss_pixel_leads = focal_loss_multiclass(
                pixel_logits, valid_pixel_lead_true
            )
        else:
            loss_pixel_leads = F.nll_loss(
                pixel_logits,
                valid_pixel_lead_true,
                ignore_index=255,
            )

        loss += loss_pixel_leads

        # Log metrics
        logs.update(
            {
                f"pixel_{stage}_loss": loss_pixel_leads,
            }
        )

        # Fusion stream
        # Compute fusion loss
        if self.config["loss"] == "wce" and stage == "train":
            loss_fuse = F.nll_loss(
                fuse_logits, true_labels, weight=self.class_weights.to(pc_logits.device)
            )
        elif self.config["loss"] == "ae" and stage == "train":
            pred_lead_fuse_labels = torch.argmax(fuse_logits, dim=1)
            correct = (pred_lead_fuse_labels.view(-1) == true_labels.view(-1)).float()
            loss_fuse = 1 - correct.mean()  # 1 - accuracy as pseudo-loss
        elif self.config["loss"] == "focal" and stage == "train":
            loss_fuse = focal_loss_multiclass(fuse_logits, true_labels)
        else:
            loss_fuse = F.nll_loss(fuse_logits, true_labels)
        loss += loss_fuse

        # Compute F1 score
        pred_lead_fuse_labels = torch.argmax(fuse_logits, dim=1)

        fuse_f1 = f1_metric(pred_lead_fuse_labels, true_labels)
        fuse_oa = oa_metric(pred_lead_fuse_labels, true_labels)

        # Log metrics
        logs.update(
            {
                f"fuse_{stage}_loss": loss_fuse,
                f"fuse_{stage}_f1": fuse_f1,
                f"fuse_{stage}_oa": fuse_oa,
            }
        )
        if stage == "val":
            self.validation_step_outputs.append(
                {"val_target": labels, "val_pred": fuse_logits}
            )

        # Compute RMSE
        rmse = torch.sqrt(loss_fuse)
        logs.update(
            {
                f"{stage}_loss": loss,
                f"{stage}_rmse": rmse,
            }
        )

        # Log all metrics
        for key, value in logs.items():
            self.log(
                key,
                value,
                on_step="loss" in key,
                on_epoch=True,
                prog_bar="val" in key,
                logger=True,
                sync_dist=True,
            )
        if stage == "test":
            return true_labels, pred_lead_fuse_labels, loss
        else:
            return loss

    def training_step(self, batch, batch_idx):
        images = batch["images"] if "images" in batch else None
        point_clouds = batch["point_cloud"] if "point_cloud" in batch else None
        labels = batch["label"]
        per_pixel_labels = (
            batch["per_pixel_labels"] if "per_pixel_labels" in batch else None
        )
        image_masks = batch["nodata_mask"] if "nodata_mask" in batch else None
        pc_feat = batch["pc_feat"] if "pc_feat" in batch else None

        loss = self.forward_and_metrics(
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
        images = batch["images"] if "images" in batch else None
        point_clouds = batch["point_cloud"] if "point_cloud" in batch else None
        labels = batch["label"]
        per_pixel_labels = (
            batch["per_pixel_labels"] if "per_pixel_labels" in batch else None
        )
        image_masks = batch["nodata_mask"] if "nodata_mask" in batch else None
        pc_feat = batch["pc_feat"] if "pc_feat" in batch else None

        loss = self.forward_and_metrics(
            images,
            image_masks,
            pc_feat,
            point_clouds,
            labels,
            per_pixel_labels,
            stage="val",
        )
        return loss

    def on_validation_epoch_end(self):
        sys_f1 = self.val_f1.compute()
        test_true = torch.cat(
            [output["val_target"] for output in self.validation_step_outputs], dim=0
        )
        test_pred = torch.cat(
            [output["val_pred"] for output in self.validation_step_outputs], dim=0
        )
        self.log("sys_f1", sys_f1, sync_dist=True)

        if sys_f1 > self.best_test_f1:
            self.best_test_f1 = sys_f1
            self.best_test_outputs = {
                "preds_all": test_pred,
                "true_labels_all": test_true,
            }

            cm = self.confmat(
                torch.argmax(test_pred, dim=1), torch.argmax(test_true, dim=1)
            )
            print(f"OA Score:{self.val_oa.compute()}")
            print(cm)

        self.validation_step_outputs.clear()
        self.val_f1.reset()
        self.val_oa.reset()

    def test_step(self, batch, batch_idx):
        images = batch["images"] if "images" in batch else None
        point_clouds = batch["point_cloud"] if "point_cloud" in batch else None
        labels = batch["label"]
        per_pixel_labels = (
            batch["per_pixel_labels"] if "per_pixel_labels" in batch else None
        )
        image_masks = batch["nodata_mask"] if "nodata_mask" in batch else None
        pc_feat = batch["pc_feat"] if "pc_feat" in batch else None

        labels, fuse_preds, loss = self.forward_and_metrics(
            images,
            image_masks,
            pc_feat,
            point_clouds,
            labels,
            per_pixel_labels,
            stage="test",
        )

        self.save_to_file(labels, fuse_preds, self.config["classes"])
        return loss

    def save_to_file(self, labels, outputs, classes):
        # Convert tensors to numpy arrays or lists as necessary
        labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
        outputs = (
            outputs.cpu().numpy() if isinstance(outputs, torch.Tensor) else outputs
        )
        num_samples = labels.shape[0]
        data = {"SampleID": np.arange(num_samples)}
        data["True"] = labels[:]
        data["Pred"] = outputs[:]

        df = pd.DataFrame(data)

        output_dir = os.path.join(
            self.config["save_dir"],
            self.config["log_name"],
            "outputs",
        )
        # Save DataFrame to a CSV file
        df.to_csv(
            os.path.join(output_dir, "test_outputs.csv"),
            mode="a",
        )

    def configure_optimizers(self):
        params = []

        # Include parameters from the image model if in 'img' or 'fuse' mode
        image_params = list(self.s2_model.parameters())
        params.append({"params": image_params, "lr": self.lr})

        # Include parameters from the point cloud model if in 'pts' or 'fuse' mode
        point_params = list(self.pc_model.parameters())
        params.append({"params": point_params, "lr": self.lr})

        # Include parameters from the fusion layers if in 'fuse' mode
        fusion_params = list(self.fuse_head.parameters())
        params.append({"params": fusion_params, "lr": self.lr})

        # Choose the optimizer based on input parameter
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                params,
                betas=(0.9, 0.999),
                eps=1e-08,
            )
        elif self.optimizer_type == "adamW":
            optimizer = torch.optim.AdamW(params)
        elif self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                params,
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
        elif self.scheduler_type == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.config["step_size"], gamma=0.1
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
