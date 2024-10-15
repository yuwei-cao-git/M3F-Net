from utils.trainer import train
import traceback
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback
import sys
import os
# local machine: wandb login --cloud --relogin

def main():
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
        "optimizer": tune.choice(["adam", "sgd", "adamW"]),
        "epochs": 100,
        "gpus": 4,
        "use_mf": tune.choice([True, False]),
        "use_residual": tune.choice([True, False]),
        "data_dir": os.path.join(os.getcwd(), "data"),
        "n_bands": 13,
        "n_classes": 13,
        "resolution": tune.choice([10, 20]),
        "scheduler": tune.choice(["plateau", "steplr", "cosine"]),
        "transforms": tune.choice([True, False]),
    }
    try:
        analysis = tune.run(
            train,
            resources_per_trial={"cpu": 8, "gpu": config.get("gpus", 1)},
            metric="val_loss",
            mode="min",
            config=config,
            num_samples=10,
            callbacks=[
                WandbLoggerCallback(
                    project="M3F-Net",
                    api_key="df8a833b419940bc3a6d3e5e04857fe61bb72eef",
                    log_config=True
                )
            ]
        )
        best_trial = analysis.get_best_trial("mean_accuracy", "max", "last")
        print(best_trial)
    except Exception as e:
        traceback.print_exc()
        raise e

if __name__ == '__main__':
    main()