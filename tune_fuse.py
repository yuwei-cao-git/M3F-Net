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
    config = {
        "mode": "fuse",  # tune.choice(["img", "pc", "fuse"]),
        "img_lr": 5e-4,  # tune.loguniform(1e-5, 1e-3),
        "pc_lr": 1e-5,  # tune.loguniform(1e-5, 1e-3),
        "fuse_lr": 0.00001,  # tune.loguniform(1e-5, 1e-3),
        "pc_loss_weight": 1.8,  # tune.loguniform(1.0, 4.0),
        "img_loss_weight": 1.3,  # tune.loguniform(1.0, 4.0),
        "fuse_loss_weight": 1.1,  # tune.loguniform(1.0, 4.0),
        "leading_loss": True,  # tune.choice([True, False]),
        "lead_loss_weight": 0.2,  # tune.loguniform(0.1, 0.5),
        "batch_size": 32,  # tune.choice([16, 32, 64]),
        "optimizer": "adamW",  # tune.choice(["adam", "sgd", "adamW"]),
        "dp_fuse": 0.7,  # tune.choice([0.3, 0.5, 0.7]),  # dropout rate
        "dp_pc": 0.5,  # tune.choice([0.3, 0.5, 0.7]),  # dropout rate
        "weighted_loss": False,  # tune.choice([True, False]),
        "train_weights": class_weights,
        "img_transforms": "compose",  # tune.choice([None, "random", "compose"]),  # augment
        "pc_transforms": True,  # tune.choice([True, False]),  # number of augmentations
        "rotate": False,  # tune.choice([True, False]),
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
        "emb_dims": 768,  # tune.choice([512, 768, 1024]),  # dimension of embeddings
        "encoder": "xl",  # tune.choice(["s", "b", "l", "xl"]),
        "linear_layers_dims": [
            1024,
            256,
        ],  # tune.choice([[1024, 256], [512, 256], [512, 128], [256, 128], [128, 128], [256, 64]]),
        "fuse_feature": True,  # tune.choice([True, False]),
        "mamba_fuse": True,  # tune.choice([True, False]),
        "fusion_dim": 128,  # tune.choice([128, 256]),
        "resolution": 20,  # tune.choice([10, 20]),
        "use_mf": False,  # tune.choice([True, False]),
        "spatial_attention": False,  # tune.choice([True, False]),
        "use_residual": False,  # tune.choice([True, False]),
        "epochs": args.max_epochs,
        "eval": tune.choice([True, False]),  # run testing
        "num_workers": args.num_workers,  # num_cpu_per_gpu
        "gpus": torch.cuda.device_count(),
        "n_samples": 2,
        "data_dir": data_dir,
        "vote": False,  # tune.choice([True, False]),
        "pc_model": "pointnext",  # tune.choise(["pointnetxt", "dgcnn"])
    }
    try:
        asha_scheduler = ASHAScheduler(max_t=1, grace_period=1, reduction_factor=2)
        trainable_with_gpu = tune.with_resources(
            train_func, {"gpu": config.get("gpus", 1)}
        )
        tuner = tune.Tuner(
            trainable_with_gpu,
            tune_config=tune.TuneConfig(
                metric="fuse_val_r2",
                mode="max",
                scheduler=asha_scheduler,
                num_samples=config["n_samples"],
            ),
            run_config=train.RunConfig(
                storage_path=config["save_dir"],
                log_to_file=("my_stdout.log", "my_stderr.log"),
                callbacks=[
                    WandbLoggerCallback(
                        project="M3F-Net-fuse-v2",
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
                results.get_best_result("fuse_val_r2", "max").config
            )
        )
    except Exception as e:
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    params = parser.parse_args()
    main(params)
