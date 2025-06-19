import argparse
import os
import torch
import numpy as np
from utils.trainer_fuse import train


def list_of_ints(arg):
    return list(map(int, arg.split(",")))


def main():
    parser = argparse.ArgumentParser(description="Train model with given parameters")

    # Basic training setup
    parser.add_argument("--mode", type=str, choices=["img", "pts", "fuse"], default="fuse")
    parser.add_argument("--task", type=str, default="regression")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to dataset directory")
    parser.add_argument("--n_classes", type=int, default=9, help="Number of classes")
    parser.add_argument("--resolution", type=int, choices=[20, 10], default=20)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size used for training")

    # Model config
    parser.add_argument("--emb_dims", type=int, default=768)
    parser.add_argument("--num_points", type=int, default=7168)
    parser.add_argument("--encoder", default="l", choices=["s", "b", "l", "xl"])
    parser.add_argument("--fusion_dim", type=int, default=128)
    parser.add_argument("--linear_layers_dims", type=list_of_ints, default=[128, 64])

    # Optimizer / LR
    parser.add_argument("--optimizer", default="adam", choices=["adam", "adamW", "sgd"])
    parser.add_argument("--scheduler", default="steplr", choices=["plateau", "steplr", "asha", "cosine"])
    parser.add_argument("--img_lr", type=float, default=5e-4)
    parser.add_argument("--pc_lr", type=float, default=1e-5)
    parser.add_argument("--fuse_lr", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--step_size", type=int, default=10)

    # Loss weights
    parser.add_argument("--pc_loss_weight", type=float, default=1.77)
    parser.add_argument("--img_loss_weight", type=float, default=1.28)
    parser.add_argument("--fuse_loss_weight", type=float, default=1.11)
    parser.add_argument("--lead_loss_weight", type=float, default=0.15)

    # Loss flags
    parser.add_argument("--leading_loss", action="store_true")
    parser.add_argument("--leading_class_weights", action="store_true")
    parser.add_argument("--weighted_loss", action="store_true")

    # Dropout and transforms
    parser.add_argument("--dp_fuse", type=float, default=0.7)
    parser.add_argument("--dp_pc", type=float, default=0.5)
    parser.add_argument("--img_transforms", default="compose", choices=["compose", "random", "None"])
    parser.add_argument("--pc_transforms", action="store_true")
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--pc_norm", action="store_true")

    # Architecture flags
    parser.add_argument("--use_mf", action="store_true", help="Use multi-fusion")
    parser.add_argument("--spatial_attention", action="store_true", help="Use spatial attention in MF fusion")
    parser.add_argument("--use_residual", action="store_true")
    parser.add_argument("--fuse_feature", action="store_true", help="Enable feature fusion")
    parser.add_argument("--mamba_fuse", action="store_true", help="Use mamba fusion")

    # System & logging
    parser.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--vote", action="store_true")
    parser.add_argument("--pc_model", default="pointnext", choices=["pointnext", "dgcnn"])
    parser.add_argument("--log_name", default="Fuse_ff_mamba_pointnext_l_Unet_20")
    parser.add_argument("--ckp", default=None)

    # Parse arguments
    params = vars(parser.parse_args())

    # Default save & data paths
    params["save_dir"] = os.path.join(os.getcwd(), "fuse_train_logs")
    params["data_dir"] = params["data_dir"] or os.path.join(os.getcwd(), "data")

    # Set classes and class weights
    if params["n_classes"] == 9:
        params["classes"] = ["BF", "BW", "CE", "LA", "PT", "PJ", "PO", "SB", "SW"]
        class_weights = [0.13429631, 0.02357711, 0.05467328, 0.04353036, 0.02462899,
            0.03230562, 0.2605792, 0.00621396, 0.42019516]
    elif params["n_classes"] == 11:
        params["classes"] = ['AB', 'PO', 'MR', 'BF', 'CE', 'PW', 'MH', 'BW', 'SW', 'OR', 'PR']
        class_weights = [0.121, 0.033, 0.045, 0.090, 0.012, 0.041,
            0.020, 0.103, 0.334, 0.010, 0.191]
    else:
        raise ValueError("Unsupported number of classes: {}".format(params["n_classes"]))

    assert len(class_weights) == params["n_classes"], "Class weights length mismatch with n_classes"
    params["train_weights"] = torch.tensor(class_weights, dtype=torch.float)

    # Ensure save directory exists
    os.makedirs(params["save_dir"], exist_ok=True)

    # Print parameters for debug
    print(params)

    # Start training
    train(params)


if __name__ == "__main__":
    main()