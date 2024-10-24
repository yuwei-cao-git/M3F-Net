import argparse
from utils.trainer import train
import os
import wandb
import torch

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Train model with given parameters")

    # Add arguments
    parser.add_argument('--mode', type=str, choices=['img', 'pts', 'both'], default='img', 
                        help="Mode to run the model: 'img', 'pts', or 'both'")
    #parser.add_argument('--data_dir', type=str, default='./data', help="path to data dir")
    parser.add_argument('--n_bands', type=int, default=12, help="number bands per tile")
    parser.add_argument('--n_classes', type=int, default=9, help="number classes")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="initial learning rate")
    parser.add_argument('--optimizer', type=str, default='adam', help="optimizer")
    parser.add_argument('--scheduler', type=str, default='steplr', help="scheduler")
    parser.add_argument('--log_name', type=str, required=True, help="Log file name")
    parser.add_argument('--resolution', type=int, choices=[20, 10], default=20, help="Resolution to use for the data")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train the model")
    parser.add_argument('--batch_size', type=int, default=4, help="Number of epochs to train the model")
    parser.add_argument('--use_mf', action='store_true', help="Use multi-fusion (set flag to enable)")
    parser.add_argument('--use_residual', action='store_true', help="Use residual connections (set flag to enable)")
    parser.add_argument('--transforms', action='store_true')
    parser.add_argument('--gpus', type=int, default=torch.cuda.device_count())

    # Parse arguments
    configs = vars(parser.parse_args())
    configs["data_dir"] = "/mnt/d/Sync/research/tree_species_estimation/tree_dataset/rmf/processed"
    configs["save_dir"] = os.path.join(os.getcwd(), "logs", configs["log_name"])
    if not os.path.exists(configs["save_dir"]):
        os.makedirs(configs["save_dir"])
    print(configs)
    wandb.init(project='M3F-Net')
    # Call the train function with parsed arguments
    train(configs)

if __name__ == "__main__":
    main()