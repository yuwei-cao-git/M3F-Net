import argparse
import os
from pathlib import Path
import torch
import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
# from torchsummary import summary
from pytorch_lightning.utilities.model_summary import ModelSummary
from models.pointNext import PointNeXtLightning
from dataset.pts import PointCloudDataModule


parser = argparse.ArgumentParser(description="pytorch-lightning parallel test")
parser.add_argument("--lr", type=float, default=0.001, help="")
parser.add_argument("--max_epochs", type=int, default=4, help="")
parser.add_argument("--batch_size", type=int, default=16, help="")
parser.add_argument("--num_workers", type=int, default=8, help="")
parser.add_argument("--data_dir", type=str, default='./data')


def main(params):
    print("Starting...")
    seed_everything(1)
    # Initialize WandB, CSV Loggers
    wandb_logger = WandbLogger(project="M3F-Net-pts")
    exp_name = params["exp_name"]
    exp_dirpath = os.path.join("checkpoints", exp_name)
    output_dir = Path(os.path.join(exp_dirpath, "output"))
    output_dir.mkdir(parents=True, exist_ok=True) 

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(exp_dirpath, "models"),  # Path to save checkpoints
        filename="{epoch}-{val_loss:.2f}",  # Filename format (epoch-val_loss)
        monitor="val_loss",  # Metric to monitor for "best" model (can be any logged metric)
        mode="min",  # Save model with the lowest "val_loss" (change to "max" for maximizing metrics)
        save_top_k=1,  # Save only the single best model based on the monitored metric
    )
    
    # Initialize the DataModule
    data_module = PointCloudDataModule(params)
    
    # initialize model
    model = PointNeXtLightning(params, in_dim=3)
    print(ModelSummary(model, max_depth=-1))  # Prints the full model summary
    # Use torchsummary to print the summary, input size should match your input data
    # summary(model, input_size=[(3, 7168), (3, 7168)])
    
    # Instantiate the Trainer
    trainer = Trainer(
        max_epochs=params["epochs"],
        logger=[wandb_logger],  # csv_logger
        callbacks=[checkpoint_callback],
        devices=params["n_gpus"], 
        accelerator="gpu"
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
        "exp_name": "pointNext_7168_WEIGHTS_AUG2",  # experiment name
        "augmentor": True,
        "batch_size": args.batch_size,  # batch size
        "train_weights": class_weights,  # training weights
        "train_path": os.path.join(args.data_dir, "rmf_laz/train"),
        "train_pickle": os.path.join(args.data_dir, "rmf_laz/train/plots_comp.pkl"),
        "val_path": os.path.join(args.data_dir, "rmf_laz/val"),
        "val_pickle": os.path.join(args.data_dir, "rmf_laz/val/plots_comp.pkl"),
        "test_path": os.path.join(args.data_dir, "rmf_laz/test"),
        "test_pickle": os.path.join(args.data_dir, "rmf_laz/test/plots_comp.pkl"),
        "augment": True,  # augment
        "n_augs": 2,  # number of augmentations
        "classes": ["BF", "BW", "CE", "LA", "PT", "PJ", "PO", "SB", "SW"],  # classes
        "n_gpus": torch.cuda.device_count(),  # number of gpus
        "epochs": args.max_epochs,  # total epochs
        "optimizer_c": "adam",  # classifier optimizer
        "scheduler": "steplr",  # classifier optimizer
        "lr_c": args.lr,  # classifier learning rate
        "adaptive_lr": True,  # adaptive learning rate
        "patience": 10,  # patience
        "step_size": 20,  # step size
        "momentum": 0.9,  # sgd momentum
        "num_points": 7168,  # number of points
        "dropout": 0.5,  # dropout rate
        "emb_dims": 1024,  # dimension of embeddings
        "k": 20,  # k nearest points
        "model_path": "",  # pretrained model path
        "cuda": True,  # use cuda
        "eval": False,  # run testing
        "num_workers": args.num_workers,  # num_cpu_per_gpu
    }

    mn = params["exp_name"]
    main(params)
