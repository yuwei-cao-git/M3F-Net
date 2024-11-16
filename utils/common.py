import yaml
import time
import wandb
import pandas as pd
import numpy as np
import os
import pytorch_lightning as pl

# get configs for a sweep from .yaml file
def get_configs_from_file(path_yaml):
    dict_yaml = yaml.load(open(path_yaml).read(), Loader=yaml.Loader)
    sweep_config = dict_yaml['sweep_config']
    params_config = dict_yaml['params_config']
    search_space = {}
    hash_keys = []
    for k,v in params_config.items():  
        search_space[k] = {"values":v}
        if len(v)>1:
            hash_keys.append(k)
        if k=='num_runs':
            assert int(v[0]) > 0
            search_space['runs'] = {"values":list(range(int(v[0])))}
    search_space['hash_keys'] = {"values":[hash_keys]}
    sweep_config['parameters'] = search_space
    return sweep_config

# modify some specific hyper parameters in sweep's config
def modify_sweep(sweep_config, dict_new):
    for key in dict_new.keys():
        sweep_config['parameters'][key] = {'values':dict_new[key]}
    return sweep_config

def GetRunTime(func):
    def call_func(*args, **kwargs):
        begin_time = time.time()
        ret = func(*args, **kwargs)
        end_time = time.time()
        Run_time = end_time - begin_time
        print("Execution time for func [%s] is [%s]"%(str(func.__name__), str(Run_time)))
        return ret
    return call_func

def get_timestamp():
    time.tzset()
    now = int(round(time.time()*1000))
    timestamp = time.strftime('%Y-%m%d-%H%M',time.localtime(now/1000))
    return timestamp

# calculate the size of a sweep's search space or the number of runs
def count_sweep(mode, entity, project, id):
    # mode: size_space, num_runs
    api = wandb.Api()
    sweep = api.sweep('%s/%s/%s'%(entity, project, id))
    if mode=='size_space':
        cnt = 1
        params= sweep.config['parameters']
        for key in params.keys():
            cnt *= len(params[key]['values'])
    elif mode=='num_runs':
        cnt = len(sweep.runs)
    return cnt

def create_comp_csv(y_true, y_pred, classes, filepath):
    """
    Create a CSV file containing true and predicted values for each class.

    Args:
        y_true (numpy.ndarray): True labels, shape (N, C).
        y_pred (numpy.ndarray): Predicted labels, shape (N, C).
        classes (list): List of class names.
        filepath (str): Path to save the CSV file.
    """
    num_samples = y_true.shape[0]
    data = {'SampleID': np.arange(num_samples)}

    # Add true and predicted values for each class
    for i, class_name in enumerate(classes):
        data[f'True_{class_name}'] = y_true[:, i]
        data[f'Pred_{class_name}'] = y_pred[:, i]

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    
def evaluate_model(sp_output_csv, pixel_output_csv, classes):
    """
    Evaluate the model's performance using the output CSV files.

    Args:
        sp_output_csv (str): Path to the superpixel output CSV file.
        pixel_output_csv (str): Path to the pixel output CSV file.
        classes (list): List of class names.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    import pandas as pd
    from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, f1_score

    # Load superpixel outputs
    sp_df = pd.read_csv(sp_output_csv)
    # Load pixel outputs
    pixel_df = pd.read_csv(sp_output_csv)

    # Compute leading species predictions and true labels
    sp_true = sp_df[[f'True_{cls}' for cls in classes]].values
    sp_pred = sp_df[[f'Pred_{cls}' for cls in classes]].values
    
    pixel_true = pixel_df[[f'True_{cls}' for cls in classes]].values
    pixel_pred = pixel_df[[f'Pred_{cls}' for cls in classes]].values

    true_leading = sp_true.argmax(axis=1)
    pred_leading = sp_pred.argmax(axis=1)
    
    pixel_true_leading = pixel_true.argmax(axis=1)
    pixel_pred_leading = pixel_pred.argmax(axis=1)

    # Compute Overall Accuracy of Leading Species
    oa_leading_species = accuracy_score(true_leading, pred_leading)
    f1_leading_species = f1_score(sp_true, sp_pred)
    wf1_leading_species = f1_score(sp_true, sp_pred, average='weighted')
    oa_pixel_leading_species = accuracy_score(pixel_true_leading, pixel_pred_leading)
    f1_pixel_leading_species = f1_score(pixel_true_leading, pixel_pred_leading)
    wf1_pixel_leading_species = f1_score(pixel_true_leading, pixel_pred_leading, average='weighted')
    
    # Compute Confusion Matrix
    cm = confusion_matrix(true_leading, pred_leading)
    pixel_cm = confusion_matrix(pixel_true_leading, pixel_pred_leading)

    # Compute overall R² and R² per Species
    all_r2 = r2_score(sp_true, sp_pred)
    all_pixel_r2 = r2_score(pixel_true, pixel_pred)
    species_r2_scores = {}
    for i, species in enumerate(classes):
        r2 = r2_score(sp_true[:, i], sp_pred[:, i])
        species_r2_scores[species] = r2
    
    pixel_species_r2_scores = {}
    for i, species in enumerate(classes):
        r2 = r2_score(pixel_true[:, i], pixel_pred[:, i])
        pixel_species_r2_scores[species] = r2

    # Compile evaluation results
    evaluation_results = {
        'Overall Accuracy of Leading Species': oa_leading_species,
        'F1 Score of Leading Species': f1_leading_species,
        'Weighted F1 Score of Leading Species': wf1_leading_species,
        'Confusion Matrix': cm,
        'Overall R2 Score': all_r2,
        'R2 Scores per Species': species_r2_scores,
        'Pixel Overall Accuracy of Leading Species': oa_pixel_leading_species,
        'Pixel F1 Score of Leading Species': f1_pixel_leading_species,
        'Pixel Weighted F1 Score of Leading Species': wf1_pixel_leading_species,
        'Pixel Confusion Matrix': pixel_cm,
        'Pixel Overall R2 Score': all_pixel_r2,
        'Pixel R2 Scores per Species': pixel_species_r2_scores
    }

    return evaluation_results

def generate_eva(model, trainer, classes, output_dir):
    if hasattr(model, 'best_test_outputs') and model.best_test_outputs is not None:
        outputs = model.best_test_outputs

        # Access the stored tensors
        preds_all = outputs['preds_all']
        true_labels_all = outputs['true_labels_all']
        pixel_preds_all = outputs['pixel_preds_all']
        true_pixel_labels_all = outputs['true_pixel_labels_all']

        # Convert tensors to NumPy arrays
        preds_all_np = preds_all.numpy()
        true_labels_all_np = true_labels_all.numpy()
        pixel_preds_all_np = pixel_preds_all.numpy()
        true_pixel_labels_all_np = true_pixel_labels_all.numpy()

        # Create CSV files
        os.makedirs(output_dir, exist_ok=True)

        # Save superpixel outputs
        sp_output_csv = os.path.join(output_dir, "best_sp_outputs.csv")
        create_comp_csv(
            y_true=true_labels_all_np,
            y_pred=preds_all_np,
            classes=classes,
            filepath=sp_output_csv
        )

        # Save pixel outputs
        pixel_output_csv = os.path.join(output_dir, "best_pixel_outputs.csv")
        create_comp_csv(
            y_true=true_pixel_labels_all_np,
            y_pred=pixel_preds_all_np,
            classes=classes,
            filepath=pixel_output_csv
        )

        # Log CSV files to wandb
        if isinstance(trainer.logger, pl.loggers.WandbLogger):
            wandb_logger = trainer.logger

            # Log the superpixel outputs CSV
            artifact_sp = wandb.Artifact(name='best_sp_outputs', type='dataset')
            artifact_sp.add_file(sp_output_csv)
            wandb_logger.experiment.log_artifact(artifact_sp)

            # Log the pixel outputs CSV
            artifact_pixel = wandb.Artifact(name='best_pixel_outputs', type='dataset')
            artifact_pixel.add_file(pixel_output_csv)
            wandb_logger.experiment.log_artifact(artifact_pixel)


        # Compute metrics using Evaluation class or function
        evaluation_results = evaluate_model(
            sp_output_csv=sp_output_csv,
            pixel_output_csv=pixel_output_csv,
            classes=classes
        )

        # Print or log evaluation results
        print("Evaluation Results:")
        print(evaluation_results)

    else:
        print("No best test outputs available.")