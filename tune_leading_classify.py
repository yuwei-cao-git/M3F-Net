import argparse
import os
import torch
import numpy as np
from utils.tunner_fuse import train_func
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.air.integrations.wandb import WandbLoggerCallback
import traceback

# Create argument parser
parser = argparse.ArgumentParser(description="Train model with given parameters")

# Add arguments
parser.add_argument("--task", type=str, default="classify")
parser.add_argument(
    "--mode",
    type=str,
    choices=["img", "pts", "fuse"],
    default="fuse",
    help="Mode to run the model: 'img', 'pts', or 'fuse'",
)
parser.add_argument("--data_dir", type=str, default=None, help="path to data dir")
parser.add_argument(
    "--max_epochs", type=int, default=100, help="Number of epochs to train the model"
)
parser.add_argument("--num_workers", type=int, default=8, help="")
parser.add_argument("--n_samples", type=int, default=30)


def main(args):
    save_dir = os.path.join(os.getcwd(), "lead_tune_logs", "ray_results")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_dir = (
        args.data_dir
        if args.data_dir is not None
        else os.path.join(os.getcwd(), "data")
    )

    config = {
        "mode": "fuse",  # tune.choice(["img", "pc", "fuse"]),
        "lr": tune.choice([1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]),
        "batch_size": tune.choice([16, 32, 64]),
        "optimizer": tune.choice(["adam", "sgd", "adamW"]),
        "dp_fuse": tune.choice([0.3, 0.5, 0.7]),  # dropout rate
        "dp_pc": tune.choice([0.3, 0.5, 0.7]),  # dropout rate
        "loss": tune.choice(["ce", "focal"]),
        "img_transforms": tune.choice([None, "random", "compose"]),  # augment
        "pc_transforms": tune.choice([True, False]),  # number of augmentations
        "rotate": tune.choice([True, False]),
        "pc_norm": True,  # tune.choice([True, False]),
        "scheduler": "asha",  # tune.choice(["plateau", "steplr", "cosine"]),
        "patience": 10,  # patience
        "step_size": 10,  # tune.choice([10, 20]), # step size
        "momentum": 0.9,  # sgd momentum
        "weight_decay": 1e-4,  # tune.choice([1e-4, 1e-6]),  # sgd momentum
        "save_dir": save_dir,
        "n_classes": 9,
        "classes": ["BF", "BW", "CE", "LA", "PT", "PJ", "PO", "SB", "SW"],  # classes
        "num_points": 7168,  # number of points
        "emb_dims": tune.choice([512, 768, 1024]),  # dimension of embeddings
        "encoder": tune.choice(["s", "b", "l", "xl"]),
        "linear_layers_dims": tune.choice(
            [
                [1024, 256],
                [1024, 128],
                [512, 256],
                [512, 128],
                [256, 128],
                [128, 128],
                [256, 64],
            ]
        ),
        "mamba_fuse": tune.choice([True, False]),
        "fusion_dim": tune.choice([128, 256]),
        "resolution": 20,  # tune.choice([10, 20]),
        "epochs": args.max_epochs,
        "num_workers": args.num_workers,  # num_cpu_per_gpu
        "gpus": torch.cuda.device_count(),
        "n_samples": args.n_samples,
        "data_dir": data_dir,
        "task": args.task,
    }
    try:
        asha_scheduler = ASHAScheduler(max_t=1, grace_period=1, reduction_factor=2)
        trainable_with_gpu = tune.with_resources(
            train_func, {"gpu": config.get("gpus", 1)}
        )
        tuner = tune.Tuner(
            trainable_with_gpu,
            tune_config=tune.TuneConfig(
                metric="sys_f1",
                mode="max",
                scheduler=asha_scheduler,
                num_samples=config["n_samples"],
            ),
            run_config=train.RunConfig(
                storage_path=config["save_dir"],
                log_to_file=("my_stdout.log", "my_stderr.log"),
                callbacks=[
                    WandbLoggerCallback(
                        project="M3F-Net-lead",
                        group="tune_group",
                        api_key=os.environ["WANDB_API_KEY"],
                        log_config=True,
                        save_checkpoints=True,
                    )
                ],
            ),
            param_space=config,
        )
        results = tuner.fit()
        print(
            "Best trial config: {}".format(
                results.get_best_result("sys_f1", "max").config
            )
        )
    except Exception as e:
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    params = parser.parse_args()
    main(params)
