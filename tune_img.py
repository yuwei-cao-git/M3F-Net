from utils.tunner_img import train_func
import traceback
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.air.integrations.wandb import WandbLoggerCallback
import os
import torch

# local machine: wandb login --cloud --relogin


def main():
    data_dir = os.path.join(os.getcwd(), "data")
    save_dir = os.path.join(os.getcwd(), "logs", "ray_results")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    config = {
        "data_dir": data_dir,
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([32, 64, 128]),
        "optimizer": tune.choice(["adam", "sgd", "adamW"]),
        "epochs": 150,
        "gpus": torch.cuda.device_count(),
        "use_mf": tune.choice([True, False]),
        "use_residual": tune.choice([True, False]),
        "n_classes": 9,
        "classes": ["BF", "BW", "CE", "LA", "PT", "PJ", "PO", "SB", "SW"],  # classes
        "resolution": tune.choice([10, 20]),
        "scheduler": "asha",  # tune.choice(["plateau", "steplr", "cosine"]),
        "transforms": tune.choice(["random", "compose"]),
        "save_dir": save_dir,
        "n_samples": 20,
    }
    try:
        # wandb.init(project='M3F-Net-ray')
        scheduler = ASHAScheduler(max_t=1, grace_period=1, reduction_factor=2)
        trainable_with_gpu = tune.with_resources(
            train_func, {"gpu": config.get("gpus", 1)}
        )
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
                        project="M3F-Net-ray",
                        group="r2_torchmetric",
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
                results.get_best_result("val_r2", "max").config
            )
        )
    except Exception as e:
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    """
    mock_api = True
    if mock_api:
        os.environ.setdefault("WANDB_MODE", "disabled")
        os.environ.setdefault("WANDB_API_KEY", "abcd")
        ray.init(
            runtime_env={"env_vars": {"WANDB_MODE": "disabled", "WANDB_API_KEY": "abcd"}}
        )
    """
    main()
