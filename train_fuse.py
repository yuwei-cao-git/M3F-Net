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
    parser.add_argument("--encoder", default="xl", choices=["s", "b", "l", "xl"])

    parser.add_argument("--fusion_dim", type=int, default=128)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "adamW", "sgd"])
    parser.add_argument(
        "--scheduler",
        default="steplr",
        choices=["plateau", "steplr", "asha", "cosine"],
    )
    parser.add_argument(
        "--img_lr", type=float, default=5e-4, help="initial learning rate"
    )
    parser.add_argument(
        "--pc_lr", type=float, default=1e-5, help="initial learning rate"
    )
    parser.add_argument(
        "--fuse_lr", type=float, default=1e-5, help="initial learning rate"
    )
    parser.add_argument("--pc_loss_weight", type=float, default=1.77)
    parser.add_argument("--img_loss_weight", type=float, default=1.28)
    parser.add_argument("--fuse_loss_weight", type=float, default=1.11)
    parser.add_argument("--leading_loss", action="store_true")
    parser.add_argument("--lead_loss_weight", type=float, default=0.19)
    parser.add_argument("--weighted_loss", default=True)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dp_fuse", type=float, default=0.7)
    parser.add_argument("--dp_pc", type=float, default=0.5)
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
    parser.add_argument("--linear_layers_dims", type=list_of_ints)
    parser.add_argument(
        "--img_transforms", default="compose", choices=["compose", "random", None]
    )
    parser.add_argument("--pc_transforms", default=True)
    parser.add_argument("--rotate", default=False)
    parser.add_argument("--pc_norm", default=True)
    parser.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--vote",
        action="store_true",
    )
    parser.add_argument(
        "--pc_model", default="pointnext", choices=["pointnext", "dgcnn"]
    )
    parser.add_argument("--log_name", default="Fuse_ff_mamba_pointnext_l_Unet_20")

    # Parse arguments
    params = vars(parser.parse_args())
    params["save_dir"] = os.path.join(os.getcwd(), "fuse_train_logs")
    params["data_dir"] = (
        params["data_dir"]
        if params["data_dir"] is not None
        else os.path.join(os.getcwd(), "data")
    )
    class_weights = [
        0.13429631,
        0.02357711,
        0.05467328,
        0.04353036,
        0.02462899,
        0.03230562,
        0.2605792,
        0.00621396,
        0.42019516,
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
