import argparse
from utils.trainer import train
import os
import torch
from utils.tunner_pts import train_func
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.air.integrations.wandb import WandbLoggerCallback
import traceback
import numpy as np

# Create argument parser
parser = argparse.ArgumentParser(description="Train model with given parameters")

# Add arguments
parser.add_argument('--mode', type=str, choices=['img', 'pts', 'both'], default='pts', 
                    help="Mode to run the model: 'img', 'pts', or 'both'")
parser.add_argument('--data_dir', type=str, default='./data', help="path to data dir")
parser.add_argument('--max_epochs', type=int, default=10, help="Number of epochs to train the model")
parser.add_argument("--num_workers", type=int, default=8, help="")

def main(args):
    save_dir = os.path.join(os.getcwd(), "pts_logs", "ray_results")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_samples = [1944, 5358, 2250, 2630, 3982, 2034, 347, 9569, 397]
    class_weights = [1 / (100 * n / 11057) for n in n_samples]
    class_weights = torch.from_numpy(np.array(class_weights)).float()
    
    config = {
        "data_dir": args.data_dir,
        "learning_rate": tune.loguniform(1e-5, 1e-1),
        "batch_size": 4, #tune.choice([32, 64, 128]),
        "optimizer": tune.choice(["adam", "sgd", "adamW"]),
        "dropout": tune.choice([0.3, 0.5, 0.7]),  # dropout rate
        "epochs": args.max_epochs,
        "augment": True,  # augment
        "n_classes": 9,
        "scheduler": "asha", # tune.choice(["plateau", "steplr", "cosine"]),
        "n_augs": tune.choice([1, 2, 3]),  # number of augmentations
        "patience": 10,  # patience
        "step_size": 20,  # step size
        "momentum": 0.9,  # sgd momentum
        "save_dir": save_dir,
        "n_samples": 20,
        "num_points": 7168,  # number of points
        "emb_dims": tune.choice([512, 768, 1024]),   # dimension of embeddings
        "train_path": os.path.join(args.data_dir, "rmf_laz/train"),
        "train_pickle": os.path.join(args.data_dir, "rmf_laz/train/plots_comp.pkl"),
        "val_path": os.path.join(args.data_dir, "rmf_laz/val"),
        "val_pickle": os.path.join(args.data_dir, "rmf_laz/val/plots_comp.pkl"),
        "test_path": os.path.join(args.data_dir, "rmf_laz/test"),
        "test_pickle": os.path.join(args.data_dir, "rmf_laz/test/plots_comp.pkl"),
        "classes": ["BF", "BW", "CE", "LA", "PT", "PJ", "PO", "SB", "SW"],  # classes
        "eval": False,  # run testing
        "num_workers": args.num_workers,  # num_cpu_per_gpu
        "gpus": torch.cuda.device_count(),
        "train_weights": class_weights,  # training weights
        "encoder": tune.choice(["s", "b", "l", 'xl']),
        "weighted_loss": tune.choice([True, False])
    }
    try:
        scheduler = ASHAScheduler(
            max_t=1,
            grace_period=1,
            reduction_factor=2)
        trainable_with_gpu = tune.with_resources(train_func, {"gpu": config.get("gpus", 1)})
        tuner = tune.Tuner(
            trainable_with_gpu,
            tune_config=tune.TuneConfig(
                metric="val_r2",
                mode="max",
                scheduler=scheduler,
                num_samples=config["n_samples"],
            ),
            run_config=train.RunConfig(
                storage_path=config["save_dir"],
                log_to_file=("my_stdout.log", "my_stderr.log"),
                callbacks=[
                    WandbLoggerCallback(
                        project="M3F-Net-pts",
                        group='tune',
                        api_key=os.environ["WANDB_API_KEY"],
                        log_config=True,
                        save_checkpoints=True,
                    )],
            ),
            param_space=config
        )
        results = tuner.fit()
        print("Best trial config: {}".format(results.get_best_result("val_r2","max").config))
    except Exception as e:
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    params = parser.parse_args()
    main(params)