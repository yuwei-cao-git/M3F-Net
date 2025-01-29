import argparse
from utils.trainer_fuse import train
import os
import torch
import numpy as np


def main():
    # Create argument parser
    # Define a custom argument type for a list of integers
    def list_of_ints(arg):
        return list(map(int, arg.split(",")))

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
        default=150,
        help="Number of epochs to train the model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Number of epochs to train the model"
    )
    parser.add_argument("--emb_dims", type=int, default=768)
    parser.add_argument("--num_points", type=int, default=7168)
    parser.add_argument("--encoder", default="b", choices=["s", "b", "l", "xl"])

    parser.add_argument("--fusion_dim", type=int, default=128)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "adamW", "sgd"])
    parser.add_argument(
        "--scheduler",
        default="steplr",
        choices=["plateau", "steplr", "asha", "cosine"],
    )
    parser.add_argument("--loss", default="ce")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dp_fuse", type=float, default=0.7)
    parser.add_argument("--dp_pc", type=float, default=0.5)
    parser.add_argument("--linear_layers_dims", type=list_of_ints)
    parser.add_argument(
        "--img_transforms", default="compose", choices=["compose", "random", "None"]
    )
    parser.add_argument("--pc_transforms", default=True)
    parser.add_argument("--rotate", default=False)
    parser.add_argument("--pc_norm", default=True)
    parser.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--log_name", default="Fuse_ff_mamba_pointnext_b_Unet_20_leading_classify"
    )
    parser.add_argument("--mamba_fuse", default=True)
    parser.add_argument("--task", default="classify")

    # Parse arguments
    params = vars(parser.parse_args())
    params["save_dir"] = os.path.join(os.getcwd(), "leading_train_logs")
    params["data_dir"] = (
        params["data_dir"]
        if params["data_dir"] is not None
        else os.path.join(os.getcwd(), "data")
    )

    if not os.path.exists(params["save_dir"]):
        os.makedirs(params["save_dir"])
    print(params)

    # Call the train function with parsed arguments
    train(params)


if __name__ == "__main__":
    main()
