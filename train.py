from utils.trainer import train
import traceback
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback
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
        "data_dir": "./data",
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
                    api_key="ubc-yuwei-cao",
                    log_config=True
                )
            ]
        )
    except Exception as e:
        traceback.print_exc()
        raise e

if __name__ == '__main__':
    main()