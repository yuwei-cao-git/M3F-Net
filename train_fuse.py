import argparse
from utils.trainer_fuse import train
import os
import torch
import numpy as np


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Train model with given parameters")

    # Add arguments
    parser.add_argument(
        "--mode",
        type=str,
        choices=["img", "pts", "fuse"],
        default="fuse",
        help="Mode to run the model: 'img', 'pts', or 'fuse'",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,  # "/mnt/d/Sync/research/tree_species_estimation/tree_dataset/rmf/processed",
        help="path to data dir",
    )
    parser.add_argument("--n_classes", type=int, default=9, help="number classes")
    parser.add_argument(
        "--classes",
        default=["BF", "BW", "CE", "LA", "PT", "PJ", "PO", "SB", "SW"],
        help="classes",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        choices=[20, 10],
        default=20,
        help="Resolution to use for the data",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=200,
        help="Number of epochs to train the model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Number of epochs to train the model"
    )
    parser.add_argument("--emb_dims", default=1024)
    parser.add_argument("--num_points", default=7168)
    parser.add_argument("--encoder", default="s", choices=["s", "b", "l", "xl"])

    parser.add_argument("--fusion_dim", default=256)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "adamW", "sgd"])
    parser.add_argument(
        "--scheduler",
        default="plateau",
        choices=["plateau", "steplr", "asha", "cosine"],
    )
    parser.add_argument(
        "--img_lr", type=float, default=0.0001, help="initial learning rate"
    )
    parser.add_argument(
        "--pc_lr", type=float, default=0.001, help="initial learning rate"
    )
    parser.add_argument(
        "--fuse_lr", type=float, default=0.001, help="initial learning rate"
    )
    parser.add_argument(
        "--pc_loss_weight", type=float, default=1.0, help="initial learning rate"
    )
    parser.add_argument(
        "--img_loss_weight", type=float, default=1.0, help="initial learning rate"
    )
    parser.add_argument(
        "--fuse_loss_weight", type=float, default=2.0, help="initial learning rate"
    )
    parser.add_argument("--leading_loss", default=True)
    parser.add_argument(
        "--lead_loss_weight", type=float, default=1.0, help="initial learning rate"
    )
    parser.add_argument("--weighted_loss", default=True)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--dp_fuse", default=0.5)
    parser.add_argument("--dp_pc", default=0.5)
    parser.add_argument(
        "--use_mf", action="store_true", help="Use multi-fusion (set flag to enable)"
    )
    parser.add_argument(
        "--spatial_attention",
        action="store_true",
        help="Use spatial attention in mf fusion",
    )
    parser.add_argument(
        "--use_residual",
        action="store_true",
        help="Use residual connections (set flag to enable)",
    )
    parser.add_argument(
        "--fuse_feature",
        default=True,
        help="feature fusion",
    )
    parser.add_argument(
        "--mamba_fuse",
        action="store_true",
        help="Use mamba fusion (set flag to enable)",
    )
    parser.add_argument(
        "--linear_layers_dims",
        default=[128, 128],
        help="dims used for the superpixels classify head",
    )
    parser.add_argument(
        "--img_transforms", default="compose", choices=["compose", "random", None]
    )
    parser.add_argument("--pc_transforms", default=True)
    parser.add_argument("--rotate", default=True)
    parser.add_argument("--pc_norm", default=True)
    parser.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--num_workers", type=int, default=8)

    # Parse arguments
    params = vars(parser.parse_args())
    params["save_dir"] = os.path.join(os.getcwd(), "fuse_train_logs")
    params["data_dir"] = (
        params.data_dir
        if params.data_dir is not None
        else os.path.join(os.getcwd(), "data")
    )
    class_weights = [
        0.02303913,
        0.13019594,
        0.05610016,
        0.07134316,
        0.12228734,
        0.08862843,
        0.01239567,
        0.48842124,
        0.00758894,
    ]
    class_weights = torch.from_numpy(np.array(class_weights)).float()
    params["train_weights"] = class_weights
    if not os.path.exists(params["save_dir"]):
        os.makedirs(params["save_dir"])
    print(params)

    # Call the train function with parsed arguments
    train(params)


if __name__ == "__main__":
    main()
