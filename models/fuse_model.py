import torch
import torch.nn as nn
import pytorch_lightning as pl

from .blocks import MF, MLPBlock, MambaFusionBlock
from .unet import UNet
from .ResUnet import ResUnet
from .pointNext import PointNextModel
from .dgcnn import DGCNN

from torchmetrics.regression import R2Score
from torchmetrics.functional import r2_score
from torchmetrics.classification import (
    MulticlassF1Score,
    ConfusionMatrix,
    MulticlassAccuracy,
)
from .loss import apply_mask, calc_loss

import os
import wandb
import pandas as pd
import numpy as np


class SuperpixelModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.use_mf = self.config["use_mf"]
        self.spatial_attention = self.config["spatial_attention"]
        self.use_residual = self.config["use_residual"]
        self.use_mamba_fuse = self.config["mamba_fuse"]
        self.fuse_feature = self.config["fuse_feature"]
        self.f = self.config["linear_layers_dims"]  # f = [512, 128]
        self.vote = self.config["vote"]
        if self.config["mode"] != "pts":
            # Initialize s2 model
            if self.config["resolution"] == 10:
                self.n_bands = 12
            else:
                self.n_bands = 9

            if self.use_mf:
                # MF Module for seasonal fusion (each season has `n_bands` channels)
                self.mf_module = MF(
                    channels=self.n_bands, spatial_att=self.spatial_attention
                )
                total_input_channels = (
                    64  # MF module outputs 64 channels after processing four seasons
                )
            else:
                total_input_channels = (
                    self.n_bands * 4
                )  # If no MF module, concatenating all seasons directly
                self.spatial_attention = False

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

        if self.config["mode"] != "img":
            # Initialize point cloud stream model
            if self.config["pc_model"] == "pointnext":
                self.pc_model = PointNextModel(
                    self.config, in_dim=6 if self.config.get("pc_norm", False) else 3
                )
            else:
                self.pc_model = DGCNN(self.config, n_classes=self.config["n_classes"])

        if self.config["mode"] == "fuse":
            # Fusion and classification layers with additional linear layer
            if self.fuse_feature:
                if self.use_mamba_fuse:
                    self.fuse_head = MambaFusionBlock(
                        in_img_chs=512,
                        in_pc_chs=(
                            self.config["emb_dims"]
                            if self.config.get("pc_model", "pointnetxt") == "pointnext"
                            else self.config["emb_dims"] * 2
                        ),
                        dim=self.config["fusion_dim"],
                        hidden_ch=self.config["linear_layers_dims"],
                        num_classes=self.config["n_classes"],
                        drop=self.config["dp_fuse"],
                    )
                else:
                    in_ch = (
                        512 + self.config["emb_dims"]
                        if self.config.get("pc_model", "pointnetxt") == "pointnext"
                        else self.config["emb_dims"] * 2
                    )
                    self.fuse_head = MLPBlock(
                        config, in_ch, self.config["linear_layers_dims"]
                    )

        # Define loss functions
        if self.config["weighted_loss"]:
            # Loss function and other parameters
            self.weights = self.config["train_weights"]  # Initialize on CPU
        self.criterion = nn.MSELoss()

        # Metrics
        self.train_r2 = R2Score()
        self.train_f1 = MulticlassF1Score(
            num_classes=self.config["n_classes"], average="weighted"
        )
        self.train_oa = MulticlassAccuracy(num_classes=self.config["n_classes"])

        self.val_r2 = R2Score()
        self.val_f1 = MulticlassF1Score(
            num_classes=self.config["n_classes"], average="weighted"
        )
        self.val_oa = MulticlassAccuracy(
            num_classes=self.config["n_classes"], average="micro"
        )

        self.test_r2 = R2Score()
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

        # Learning rates for different parts
        self.img_lr = self.config.get("img_lr")
        self.pc_lr = self.config.get("pc_lr")
        self.fusion_lr = self.config.get("fuse_lr")

        # Loss weights
        self.pc_loss_weight = self.config.get("pc_loss_weight", 2.0)
        self.img_loss_weight = self.config.get("img_loss_weight", 1.0)
        self.fuse_loss_weight = self.config.get("fuse_loss_weight", 1.0)
        if self.config["leading_loss"]:
            self.leading_loss_weight = self.config.get("lead_loss_weight", 0.1)

        self.best_test_r2 = 0.0
        self.best_test_outputs = None
        self.validation_step_outputs = []

    def forward(self, images, pc_feat, xyz):
        image_outputs = None
        img_emb = None
        point_outputs = None
        pc_emb = None

        if self.config["mode"] != "pts":
            # Process images
            if self.use_mf:
                fused_features = self.mf_module(images)
            else:
                batch_size, num_seasons, num_channels, width, height = images.shape
                fused_features = images.view(
                    batch_size, num_seasons * num_channels, width, height
                )
            image_outputs, img_emb = self.s2_model(fused_features)

        if self.config["mode"] != "img":
            # Process point clouds
            if self.config["pc_model"] == "pointnext":
                point_outputs, pc_emb = self.pc_model(pc_feat, xyz)
                pc_emb = torch.max(pc_emb, dim=2)[0]  # Shape: (batch_size, feature_dim)
            else:
                point_outputs, pc_emb = self.pc_model(torch.cat([pc_feat, xyz], dim=1))

        if self.config["mode"] == "fuse":
            if self.fuse_feature:
                # Fusion and classification
                class_output = self.fuse_head(img_emb, pc_emb)
                return image_outputs, point_outputs, class_output
            else:
                return image_outputs, point_outputs
        elif self.config["mode"] == "img":
            return image_outputs
        else:
            return point_outputs

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
        if self.config["mode"] == "fuse":
            if self.fuse_feature:
                pixel_preds, pc_preds, fuse_preds = self.forward(
                    images, pc_feat, point_clouds
                )
            else:
                pixel_preds, pc_preds = self.forward(images, pc_feat, point_clouds)
                fuse_preds = None
        elif self.config["mode"] == "img":
            pixel_preds = self.forward(images, None, None)
            pc_preds = None
            fuse_preds = None
        else:
            pc_preds = self.forward(None, pc_feat=pc_feat, xyz=point_clouds)
            pixel_preds = None
            fuse_preds = None

        true_labels = torch.argmax(labels, dim=1)
        loss = 0
        logs = {}

        # Select appropriate metric instances based on the stage
        if stage == "train":
            r2_metric = self.train_r2
            f1_metric = self.train_f1
            oa_metric = self.train_oa
        elif stage == "val":
            r2_metric = self.val_r2
            f1_metric = self.val_f1
            oa_metric = self.val_oa
        else:  # stage == "test"
            r2_metric = self.test_r2
            f1_metric = self.test_f1
            oa_metric = self.test_oa

        # Point cloud stream
        if self.config["mode"] != "img":
            # Compute point cloud loss
            if self.config["weighted_loss"] and stage == "train":
                self.weights = self.weights.to(pc_preds.device)
                loss_point = calc_loss(labels, pc_preds, self.weights)
            else:
                loss_point = self.criterion(pc_preds, labels)
            loss += self.pc_loss_weight * loss_point

            # Compute R² metric
            pc_preds_rounded = torch.round(pc_preds, decimals=2)
            pc_r2 = r2_metric(pc_preds_rounded.view(-1), labels.view(-1))

            # Compute F1 score
            pred_lead_pc_labels = torch.argmax(pc_preds, dim=1)

            if self.config["leading_loss"] and stage == "train":
                incorrect = (
                    pred_lead_pc_labels.view(-1) != true_labels.view(-1)
                ).float()
                if self.config["leading_class_weights"]:
                    incorrect = incorrect.squeeze()
                    class_weights = torch.tensor(
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
                    ).to(true_labels.device)
                    weights = class_weights[true_labels.squeeze()]
                    loss_leads = (incorrect * weights).mean()
                else:
                    loss_leads = incorrect.mean()
                loss += self.leading_loss_weight * loss_leads

            # Log metrics
            logs.update(
                {
                    f"pc_{stage}_loss": loss_point,
                    f"pc_{stage}_r2": pc_r2,
                }
            )

        # Image stream
        if self.config["mode"] != "pts":
            # Apply mask to predictions and labels
            valid_pixel_preds, valid_pixel_true = apply_mask(
                pixel_preds, pixel_labels, img_masks
            )

            # Compute pixel-level loss
            # loss_pixel = self.criterion(valid_pixel_preds, valid_pixel_true)
            if self.config["weighted_loss"] and stage == "train":
                self.weights = self.weights.to(pc_preds.device)
                loss_pixel = calc_loss(
                    valid_pixel_true, valid_pixel_preds, self.weights
                )
            else:
                loss_pixel = self.criterion(valid_pixel_preds, valid_pixel_true)
            loss += self.img_loss_weight * loss_pixel

            # Compute R² metric
            valid_pixel_preds_rounded = torch.round(valid_pixel_preds, decimals=2)
            pixel_r2 = r2_metric(
                valid_pixel_preds_rounded.view(-1), valid_pixel_true.view(-1)
            )

            pred_lead_pixel_labels = torch.argmax(pixel_preds, dim=1)
            true_lead_pixel_labels = torch.argmax(pixel_labels, dim=1)
            valid_pixel_lead_preds, valid_pixel_lead_true = apply_mask(
                pred_lead_pixel_labels,
                true_lead_pixel_labels,
                img_masks,
                multi_class=False,
                keep_shp=False,
            )
            # print(valid_pixel_lead_preds.shape) # torch.Size([16, 128, 128])

            if self.config["leading_loss"] and stage == "train":
                correct = (
                    valid_pixel_lead_preds.view(-1) == valid_pixel_lead_true.view(-1)
                ).float()
                loss_pixel_leads = 1 - correct.mean()  # 1 - accuracy as pseudo-loss
                loss += self.leading_loss_weight * loss_pixel_leads

            # Log metrics
            logs.update(
                {
                    f"pixel_{stage}_loss": loss_pixel,
                    f"pixel_{stage}_r2": pixel_r2,
                }
            )

        # Fusion stream
        if self.config["mode"] == "fuse":
            if self.config.get("fuse_feature", False):
                # Compute fusion loss
                if self.config["weighted_loss"] and stage == "train":
                    loss_fuse = calc_loss(labels, fuse_preds, self.weights)
                else:
                    loss_fuse = self.criterion(fuse_preds, labels)
                loss += self.fuse_loss_weight * loss_fuse

                # Compute R² metric
                fuse_preds_rounded = torch.round(fuse_preds, decimals=2)
                fuse_r2 = r2_metric(fuse_preds_rounded.view(-1), labels.view(-1))

                # Compute F1 score
                pred_lead_fuse_labels = torch.argmax(fuse_preds, dim=1)

                fuse_f1 = f1_metric(pred_lead_fuse_labels, true_labels)
                fuse_oa = oa_metric(pred_lead_fuse_labels, true_labels)

                # Log metrics
                logs.update(
                    {
                        f"fuse_{stage}_loss": loss_fuse,
                        f"fuse_{stage}_r2": fuse_r2,
                        f"fuse_{stage}_f1": fuse_f1,
                        f"fuse_{stage}_oa": fuse_oa,
                    }
                )
                if stage == "val":
                    self.validation_step_outputs.append(
                        {"val_target": labels, "val_pred": fuse_preds}
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
                prog_bar="val_r2" in key,
                logger=True,
                sync_dist=True,
            )
        if stage == "test":
            return labels, fuse_preds, loss
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
        sys_r2 = self.val_r2.compute()
        test_true = torch.cat(
            [output["val_target"] for output in self.validation_step_outputs], dim=0
        )
        test_pred = torch.cat(
            [output["val_pred"] for output in self.validation_step_outputs], dim=0
        )

        last_epoch_val_r2 = r2_score(
            torch.round(test_pred.flatten(), decimals=1), test_true.flatten()
        )
        self.log("ave_val_r2", last_epoch_val_r2, sync_dist=True)
        self.log("sys_r2", sys_r2, sync_dist=True)

        print(f"average r2 score at epoch {self.current_epoch}: {last_epoch_val_r2}")
        if last_epoch_val_r2 > self.best_test_r2:
            self.best_test_r2 = last_epoch_val_r2
            self.best_test_outputs = {
                "preds_all": test_pred,
                "true_labels_all": test_true,
            }

            cm = self.confmat(
                torch.argmax(test_pred, dim=1), torch.argmax(test_true, dim=1)
            )

            print(f"F1 Score:{self.val_f1.compute()}")
            print(f"OA Score:{self.val_oa.compute()}")
            print("Confusion Matrix at best R²:")
            print(cm)
            print(f"R2 Score:{sys_r2}")
            print(
                f"r2_score per class check: {r2_score(torch.round(test_pred, decimals=1), test_true, multioutput='raw_values')}"
            )
            """
            output_dir = os.path.join(
                self.config["save_dir"],
                "outputs",
            )
            # _ = generate_eva(self.best_test_outputs, self.config["classes"], output_dir)
            
            cm_array = cm.cpu().numpy()

            # Convert to Pandas DataFrame and add species names
            import pandas as pd

            species_names = self.config["classes"]
            cm_df = pd.DataFrame(cm_array, index=species_names, columns=species_names)

            # Log the confusion matrix as a WandB Table
            cm_table = wandb.Table(dataframe=cm_df)
            self.logger.experiment.log(
                {f"confusion_matrix at epoch {self.current_epoch}": cm_table}
            )
            """
        self.validation_step_outputs.clear()
        self.val_r2.reset()
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

        # Add true and predicted values for each class
        for i, class_name in enumerate(classes):
            data[f"True_{class_name}"] = labels[:, i]
            data[f"Pred_{class_name}"] = outputs[:, i]

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
        if self.config["mode"] != "pts":
            image_params = list(self.s2_model.parameters())
            if self.use_mf:
                image_params += list(self.mf_module.parameters())
            params.append({"params": image_params, "lr": self.img_lr})

        # Include parameters from the point cloud model if in 'pts' or 'fuse' mode
        if self.config["mode"] != "img":
            point_params = list(self.pc_model.parameters())
            params.append({"params": point_params, "lr": self.pc_lr})

        # Include parameters from the fusion layers if in 'fuse' mode
        if self.config["mode"] == "fuse":
            if self.config.get("fuse_feature", False):
                fusion_params = list(self.fuse_head.parameters())
                params.append({"params": fusion_params, "lr": self.fusion_lr})
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
