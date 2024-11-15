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
parser.add_argument(
    "--mode",
    type=str,
    choices=["img", "pts", "fuse"],
    default="fuse",
    help="Mode to run the model: 'img', 'pts', or 'fuse'",
)
parser.add_argument("--data_dir", type=str, default=None, help="path to data dir")
parser.add_argument(
    "--max_epochs", type=int, default=10, help="Number of epochs to train the model"
)
parser.add_argument("--num_workers", type=int, default=8, help="")


def main(args):
    save_dir = os.path.join(os.getcwd(), "fuse_tune_logs", "ray_results")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_dir = (
        args.data_dir
        if args.data_dir is not None
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
    config = {
        "mode": "fuse",  # tune.choice(["img", "pc", "fuse"]),
        "img_lr": tune.loguniform(1e-5, 1e-2),
        "pc_lr": tune.loguniform(1e-5, 1e-2),
        "fuse_lr": tune.loguniform(1e-5, 1e-2),
        "pc_loss_weight": 2.0,  # tune.loguniform(1.0, 4.0),
        "img_loss_weight": 1.0,  # tune.loguniform(1.0, 4.0),
        "fuse_loss_weight": 1.0,  # tune.loguniform(1.0, 4.0),
        "batch_size": 4,  # tune.choice([16, 32, 64, 128]),
        "optimizer": "adam",  # tune.choice(["adam", "sgd", "adamW"]),
        "dropout": 0.5,  # tune.choice([0.3, 0.5, 0.7]),  # dropout rate
        "weighted_loss": True,  # tune.choice([True, False]),
        "train_weights": class_weights,
        "img_transforms": "compose",  # tune.choice([None, "random", "compose"]),  # augment
        "pc_transforms": True,  # tune.choice([True, False]),  # number of augmentations
        "rotate": True,  # tune.choice([True, False]),
        "pc_norm": True,  # tune.choice([True, False]),
        "scheduler": "plateau",  # "asha",  # tune.choice(["plateau", "steplr", "cosine"]),
        "patience": 10,  # patience
        "step_size": 10,  # tune.choice([10, 20]), # step size
        "momentum": 0.9,  # sgd momentum
        "weight_decay": tune.choice([1e-4, 1e-6]),  # sgd momentum
        "save_dir": save_dir,
        "n_classes": 9,
        "classes": ["BF", "BW", "CE", "LA", "PT", "PJ", "PO", "SB", "SW"],  # classes
        "num_points": 7168,  # number of points
        "emb_dims": tune.choice([512, 768, 1024]),  # dimension of embeddings
        "encoder": tune.choice(["s", "b", "l", "xl"]),
        "linear_layers_dims": tune.choice(
            [[1024, 256], [512, 128], [256, 128], [128, 128], [256, 64]]
        ),
        "fuse_feature": tune.choice([True, False]),
        "mamba_fuse": tune.choice([True, False]),
        "fusion_dim": tune.choice([128, 256]),
        "resolution": 20,  # tune.choice([10, 20]),
        "use_mf": tune.choice([True, False]),
        "spatial_attention": tune.choice([True, False]),
        "use_residual": tune.choice([True, False]),
        "epochs": args.max_epochs,
        "eval": True,  # run testing
        "num_workers": args.num_workers,  # num_cpu_per_gpu
        "gpus": torch.cuda.device_count(),
        "n_samples": 20,
        "data_dir": data_dir,
    }
    try:
        # asha_scheduler = ASHAScheduler(max_t=1, grace_period=1, reduction_factor=2)
        trainable_with_gpu = tune.with_resources(
            train_func, {"gpu": config.get("gpus", 1)}
        )
        tuner = tune.Tuner(
            trainable_with_gpu,
            tune_config=tune.TuneConfig(
                metric="fuse_val_r2",
                mode="max",
                # scheduler=asha_scheduler,
                num_samples=config["n_samples"],
            ),
            run_config=train.RunConfig(
                storage_path=config["save_dir"],
                log_to_file=("my_stdout.log", "my_stderr.log"),
                callbacks=[
                    WandbLoggerCallback(
                        project="M3F-Net-fuse",
                        group="tune_v4",
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
                results.get_best_result("fuse_val_r2", "max").config
            )
        )
    except Exception as e:
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    params = parser.parse_args()
    main(params)
