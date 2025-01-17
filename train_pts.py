import argparse
import os
from pathlib import Path
import torch
import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

# from torchsummary import summary
# from pytorch_lightning.utilities.model_summary import ModelSummary
from models.pc_model import PointNeXtLightning
from dataset.pts import PointCloudDataModule


parser = argparse.ArgumentParser(description="pytorch-lightning parallel test")
parser.add_argument("--lr", type=float, default=0.0005, help="")
parser.add_argument("--max_epochs", type=int, default=250, help="")
parser.add_argument("--batch_size", type=int, default=8, help="")
parser.add_argument("--num_workers", type=int, default=8, help="")
parser.add_argument("--data_dir", type=str, default="./data")


def main(params):
    print("Starting...")
    seed_everything(1)
    # Initialize WandB, CSV Loggers
    wandb_logger = WandbLogger(project="M3F-Net-pts")
    exp_name = params["exp_name"]
    exp_dirpath = os.path.join("pts_logs", exp_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(exp_dirpath, "models"),  # Path to save checkpoints
        filename="final_model",  # Filename format (epoch-val_loss)
        monitor="val_loss",  # Metric to monitor for "best" model (can be any logged metric)
        mode="min",  # Save model with the lowest "val_loss" (change to "max" for maximizing metrics)
        save_top_k=1,  # Save only the single best model based on the monitored metric
    )

    # Initialize the DataModule
    data_module = PointCloudDataModule(params)

    # initialize model
    model = PointNeXtLightning(params, in_dim=3)
    # print(ModelSummary(model, max_depth=-1))  # Prints the full model summary
    # Use torchsummary to print the summary, input size should match your input data
    # summary(model, input_size=[(3, 7168), (3, 7168)])

    # Instantiate the Trainer
    trainer = Trainer(
        max_epochs=params["epochs"],
        logger=[wandb_logger],  # csv_logger
        callbacks=[checkpoint_callback],
        devices=params["n_gpus"],
    )

    trainer.fit(model, data_module)

    if params["eval"]:
        trainer.test(model, data_module)


if __name__ == "__main__":
    n_samples = [1944, 5358, 2250, 2630, 3982, 2034, 347, 9569, 397]
    class_weights = [1 / (100 * n / 11057) for n in n_samples]
    class_weights = torch.from_numpy(np.array(class_weights)).float()
    args = parser.parse_args()
    params = {
        "mode": "pc",
        "exp_name": "pointNext_7168_WEIGHTS",  # experiment name
        "batch_size": args.batch_size,  # batch size
        "train_weights": class_weights,  # training weights
        "train_path": os.path.join(args.data_dir, "rmf_laz/train"),
        "train_pickle": os.path.join(args.data_dir, "rmf_laz/train/plots_comp.pkl"),
        "val_path": os.path.join(args.data_dir, "rmf_laz/val"),
        "val_pickle": os.path.join(args.data_dir, "rmf_laz/val/plots_comp.pkl"),
        "test_path": os.path.join(args.data_dir, "rmf_laz/test"),
        "test_pickle": os.path.join(args.data_dir, "rmf_laz/test/plots_comp.pkl"),
        "augment": True,  # augment
        "n_augs": 1,  # number of augmentations
        "classes": ["BF", "BW", "CE", "LA", "PT", "PJ", "PO", "SB", "SW"],  # classes
        "n_classes": 9,
        "n_gpus": torch.cuda.device_count(),  # number of gpus
        "epochs": args.max_epochs,  # total epochs
        "optimizer": "adamW",  # classifier optimizer
        "scheduler": "cosine",  # classifier optimizer
        "learning_rate": args.lr,  # classifier learning rate
        "patience": 10,  # patience
        "step_size": 20,  # step size
        "momentum": 0.9,  # sgd momentum
        "num_points": 7168,  # number of points
        "dp_pc": 0.3,  # dropout rate
        "emb_dims": 768,  # dimension of embeddings
        "weighted_loss": False,  # pretrained model path
        "eval": True,  # run testing
        "num_workers": args.num_workers,  # num_cpu_per_gpu
        "encoder": "l",
        "pc_norm": False,
    }

    mn = params["exp_name"]
    main(params)
