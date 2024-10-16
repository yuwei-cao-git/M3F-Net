from utils.trainer import train_func
import traceback
import wandb
import ray
from ray import tune, train
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
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
        "n_bands": 13,
        "n_classes": 13,
        "resolution": tune.choice([10, 20]),
        "scheduler": tune.choice(["plateau", "steplr", "cosine"]),
        "transforms": tune.choice([True, False]),
    }
    try:
        wandb.init(project='M3F-Net')
        trainable_with_resources = tune.with_resources(train_func,
        resources_per_trial={"cpu": 32, "gpu": 4})
        tuner = tune.Tuner(
            trainable_with_resources,
            tune_config=tune.TuneConfig(
                metric="val_loss",
                mode="min",
                num_samples=10,
            ),
            run_config=train.RunConfig(
                storage_path="~/scratch/ray_results",
                log_to_file=("my_stdout.log", "my_stderr.log"),
                callbacks=[
                    WandbLoggerCallback(
                        project="M3F-Net",
                        log_config=True,
                        save_checkpoints=True
                    )],
            ),
            param_space=config
        )
        results = tuner.fit()
        print("the best config is:" + str(results.get_best_result().config))
    except Exception as e:
        traceback.print_exc()
        raise e

if __name__ == '__main__':
    mock_api = True

    if mock_api:
        os.environ.setdefault("WANDB_MODE", "disabled")
        os.environ.setdefault("WANDB_API_KEY", "abcd")
        ray.init(
            runtime_env={"env_vars": {"WANDB_MODE": "disabled", "WANDB_API_KEY": "abcd"}}
        )
    main()