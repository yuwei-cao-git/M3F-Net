import os
import time

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from models.pointNext import PointNeXtLightning
from dataset.pts import PointCloudDataModule

import wandb
from ray import tune, train

def train_func(params):
    seed_everything(1, workers=True)
    # Initialize WandB, CSV Loggers
    wandb_logger = WandbLogger(project="M3F-Net-pts", name=f"trial_{tune.Trainable().trial_id}", save_dir=params["save_dir"], log_model=True)

    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{val_loss:.2f}",  # Filename format (epoch-val_loss)
        monitor="val_loss",  # Metric to monitor for "best" model (can be any logged metric)
        mode="min",  # Save model with the lowest "val_loss" (change to "max" for maximizing metrics)
        save_top_k=1,  # Save only the single best model based on the monitored metric
    )
    
    # Initialize the DataModule
    data_module = PointCloudDataModule(params)
    
    # initialize model
    model = PointNeXtLightning(params, in_dim=3)
    # print(ModelSummary(model, max_depth=-1))  # Prints the full model summary
    # Use torchsummary to print the summary, input size should match your input data
    # summary(model, input_size=[(3, 7168), (3, 7168)])
    
    # Instantiate the Trainer
    trainer = Trainer(
        max_epochs=params["epochs"],
        logger=[wandb_logger],  # csv_logger
        callbacks=[checkpoint_callback],
        devices=params["n_gpus"], 
        strategy='ddp'
    )
    
    trainer.fit(model, data_module)
    
    # Report the final metric to Ray Tune
    final_result = trainer.callback_metrics["val_r2"].item()
    train.report({"val_r2": final_result})

    # Save the best model after training
    trainer.save_checkpoint(os.path.join(params["save_dir"], f"trial_{tune.Trainable().trial_id}", "final_model.pt"))
    

    if params["eval"]:
        trainer.test(model, data_module)
    
    time.sleep(5)  # Wait for wandb to finish logging
    wandb.finish() 