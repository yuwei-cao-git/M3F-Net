import argparse
from utils.trainer_img import train
import os
import wandb
import torch


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Train model with given parameters")

    # Add arguments
    parser.add_argument("--mode", type=str, choices=["img", "pts", "fuse"], default="fuse", help="Mode to run the model: 'img', 'pts', or 'fuse'")
    parser.add_argument("--data_dir", type=str, default="/mnt/d/Sync/research/tree_species_estimation/tree_dataset/rmf/processed", help="path to data dir")
    parser.add_argument("--n_classes", type=int, default=9, help="number classes")
    parser.add_argument("--log_name", type=str, required=True, help="Log file name")
    parser.add_argument("--resolution", type=int, choices=[20, 10], default=20, help="Resolution to use for the data")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the model")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of epochs to train the model")
    parser.add_argument("--emb_dims", default=1024)
    parser.add_argument("--num_points", default=7168)
    parser.add_argument("--encoder", default="s", choices=["s", "b", "l", "xl"])
    parser.add_argument("--optimizer", default="adam", choices=["adam", "adamW", "sgd"])
    parser.add_argument("--scheduler", default="plateau", choices=["plateau", "steplr", "asha", "cosine"])
    parser.add_argument("--learning_rate", type=float, default=0.001, help="initial learning rate")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--step_size", type=int, default=20)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", default=0.5)
    parser.add_argument("--use_mf", action="store_true", help="Use multi-fusion (set flag to enable)")
    parser.add_argument("--use_residual", action="store_true", help="Use residual connections (set flag to enable)",)
    parser.add_argument("--img_transforms", action="store_true")
    parser.add_argument("--pc_transforms", action="store_true")
    parser.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    

    # Parse arguments
    params = vars(parser.parse_args())
    params["data_dir"] = os.path.join(params["data_dir"], f"str({params["resolution"]})m", "fusion")
    print(params)

    # Call the train function with parsed arguments
    train(params)


if __name__ == "__main__":
    main()
