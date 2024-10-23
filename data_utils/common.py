import glob
import os
from pathlib import Path
from itertools import cycle, islice

import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
#from plyer import notification
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset


def read_las(pointcloudfile, get_attributes=False, useevery=1):
    """
    :param pointcloudfile: specification of input file (format: las or laz)
    :param get_attributes: if True, will return all attributes in file, otherwise will only return XYZ (default is False)
    :param useevery: value specifies every n-th point to use from input, i.e. simple subsampling (default is 1, i.e. returning every point)
    :return: 3D array of points (x,y,z) of length number of points in input file (or subsampled by 'useevery')
    """

    # Read file
    inFile = laspy.read(pointcloudfile)

    # Get coordinates (XYZ)
    coords = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
    coords = coords[::useevery, :]

    # Return coordinates only
    if get_attributes == False:
        return coords

    # Return coordinates and attributes
    else:
        las_fields = [info.name for info in inFile.points.point_format.dimensions]
        attributes = {}
        # for las_field in las_fields[3:]:  # skip the X,Y,Z fields
        for las_field in las_fields:  # get all fields
            attributes[las_field] = inFile.points[las_field][::useevery]
        return (coords, attributes)


class PointCloudsInPickle(Dataset):
    """Point cloud dataset where one data point is a file."""

    def __init__(
        self,
        filepath,
        pickle,
        # column_name="",
        # max_points=200_000,
        # samp_meth=None,  # one of None, "fps", or "random"
        # use_columns=None,
    ):
        """
        Args:
            pickle (string): Path to pickle dataframe
            column_name (string): Column name to use as target variable (e.g. "Classification")
            use_columns (list[string]): Column names to add as additional input
        """
        self.filepath = filepath
        self.pickle = pd.read_pickle(pickle)
        # self.column_name = column_name
        # self.max_points = max_points
        # if use_columns is None:
        #     use_columns = []
        # self.use_columns = use_columns
        # self.samp_meth = samp_meth
        super().__init__()

    def __len__(self):
        return len(self.pickle)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get file name
        pickle_idx = self.pickle.iloc[idx : idx + 1]
        filename = pickle_idx["FileName"].item()

        # Get file path
        file = os.path.join(self.filepath, filename)

        # Read las/laz file
        coords = read_las(file, get_attributes=False)

        xyz = coords - np.mean(coords, axis=0)  # centralize coordinates

        # impute target
        target = pickle_idx["perc_specs"].item()
        target = target.replace("[", "")
        target = target.replace("]", "")
        target = target.split(",")
        target = [float(i) for i in target]  # convert items in target to float

        coords = torch.from_numpy(coords).float()
        target = torch.from_numpy(np.array(target)).type(torch.FloatTensor)
        # target = torch.from_numpy(np.array(target)).half()
        if coords.shape[0] < 100:
            return None
        return coords, xyz, target


def write_las(outpoints, outfilepath, attribute_dict={}):
    """
    :param outpoints: 3D array of points to be written to output file
    :param outfilepath: specification of output file (format: las or laz)
    :param attribute_dict: dictionary of attributes (key: name of attribute; value: 1D array of attribute values in order of points in 'outpoints'); if not specified, dictionary is empty and nothing is added
    :return: None
    """
    import laspy

    hdr = laspy.LasHeader(version="1.4", point_format=6)
    hdr.x_scale = 0.00025
    hdr.y_scale = 0.00025
    hdr.z_scale = 0.00025
    mean_extent = np.mean(outpoints, axis=0)
    hdr.x_offset = int(mean_extent[0])
    hdr.y_offset = int(mean_extent[1])
    hdr.z_offset = int(mean_extent[2])

    las = laspy.LasData(hdr)

    las.x = outpoints[0]
    las.y = outpoints[1]
    las.z = outpoints[2]
    for key, vals in attribute_dict.items():
        try:
            las[key] = vals
        except:
            las.add_extra_dim(laspy.ExtraBytesParams(name=key, type=type(vals[0])))
            las[key] = vals

    las.write(outfilepath)
    
class IOStream:
    # Adapted from https://github.com/vinits5/learning3d/blob/master/examples/train_pointnet.py
    def __init__(self, path):
        # Open file in append
        self.f = open(path, "a")

    def cprint(self, text):
        # Print and write text to file
        print(text)  # print text
        self.f.write(text + "\n")  # write text and new line
        self.f.flush  # flush file

    def close(self):
        self.f.close()  # close file


def _init_(model_name):
    # Create folder structure
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    if not os.path.exists("checkpoints/" + model_name):
        os.makedirs("checkpoints/" + model_name)
    if not os.path.exists("checkpoints/" + model_name + "/models"):
        os.makedirs("checkpoints/" + model_name + "/models")
    if not os.path.exists("checkpoints/" + model_name + "/output"):
        os.makedirs("checkpoints/" + model_name + "/output")
    if not os.path.exists("checkpoints/" + model_name + "/output/laz"):
        os.makedirs("checkpoints/" + model_name + "/output/laz")
    
   
def check_multi_gpu(n_gpus):
    # Check if multiple GPUs are available
    if torch.cuda.device_count() > 1 and n_gpus > 1:
        multi_gpu = True
        print("Using Multiple GPUs")
    else:
        multi_gpu = False
        
    return multi_gpu


def create_empty_df():
    # Create an empty dataframe with specific dtype
    df = pd.DataFrame(
        {
            "Model": pd.Series(dtype="str"),
            "Point Density": pd.Series(dtype="str"),
            "Overall Accuracy": pd.Series(dtype="float"),
            "F1": pd.Series(dtype="float"),
            "Augmentation": pd.Series(dtype="str"),
            "Sampling Method": pd.Series(dtype="str"),
        }
    )

    return df


def variable_df(variables, col_names):
    # Create a dataframe from list of variables
    df = pd.DataFrame([variables], columns=col_names)
    
    return df


def concat_df(df_list):
    # Concate multiple dataframes in list
    df = pd.concat(df_list, ignore_index=True)
    return df

'''
def notifi(title, message):
    # Creates a pop-up notification
    notification.notify(title=title, message=message, timeout=10)
'''
    
    
def create_comp_csv(y_true, y_pred, classes, filepath):
    # Create a CSV of the true and predicted species proportions
    classes = cycle(classes) # cycle classes
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}) # create dataframe
    df["SpeciesName"] = list(islice(classes, len(df))) # repeat list of classes
    species = df.pop("SpeciesName") # remove species name
    df.insert(0, "SpeciesName", species) # add species name
    df.to_csv(filepath, index=False) # save to csv
    
    
def get_stats(df):
    # Get stats
    r2 = r2_score(df["y_true"], df["y_pred"])  # r2
    rmse = np.sqrt(mean_squared_error(df["y_true"], df["y_pred"]))  # rmse
    df_max = np.max(df["Difference"])  # max difference
    df_min = np.min(df["Difference"])  # min difference
    df_std = np.std(df["Difference"])  # std difference
    df_var = np.var(df["Difference"])  # var difference
    df_count = np.count_nonzero(df["y_true"])  # count of none 0 y_true

    return pd.Series(
        dict(
            r2=r2,
            rmse=rmse,
            maximum=df_max,
            minimum=df_min,
            std=df_std,
            var=df_var,
            count=df_count,
        )
    )


def get_df_stats(csv, min_dif, max_dif):
    # Create dataframe
    df = pd.read_csv(csv)  # read in csv
    df["Difference"] = df["y_pred"] - df["y_true"]  # difference of y_pred and y_true
    df["Sum"] = df["y_pred"] + df["y_true"]  # sum of y_pred and y_true
    df["Between"] = df["Difference"].between(min_dif, max_dif)  # boolean of Difference

    # Print number of True and False values of min and max difference values
    print("Count")
    print(df.groupby("Between")["Difference"].count())
    print()

    # Calculate and print get_stats fpr df
    print("Stats")
    df_stats = df.groupby("SpeciesName").apply(get_stats)
    print(df_stats)
    print("#################################################")
    print()

    return df_stats


def scatter_plot(csv, point_density, root_dir):
    df = pd.read_csv(csv)
    species = df.SpeciesName.unique()
    for i in species:
        species_csv = df[df.SpeciesName == i]
        species_csv.plot.scatter(x="y_pred", y="y_true")
        plt.title(f"{i}: {point_density}")
        plt.savefig(os.path.join(root_dir, f"{i}_{point_density}.png"))
        plt.close()


def plot_stats(
    root_dir,
    point_densities,
    model,
    stats=["r2", "rmse"],
    save_csv=False,
    csv_name=None,
    save_fig=False,
):
    # Set Plot Parameters
    plt.rcParams["figure.figsize"] = [15.00, 7.00]  # figure size
    plt.rcParams["figure.autolayout"] = True  # auto layout
    plt.rcParams["figure.facecolor"] = "white"  # facecolor

    dfs_r2 = []
    dfs_rmse = []
    for x in point_densities:
        # Print point density
        print(f"Point Density: {str(x)}")

        # Get root directory
        model_output = os.path.join(root_dir, f"{model}_{x}\output")

        # Get CSVs
        csv = list(Path(model_output).glob("outputs*.csv"))

        # Iterate through CSVs
        for y in csv:
            # Create scatter plots
            scatter_plot(y, x, model_output)

            # Calculate stats
            csv_stats = get_df_stats(y, -0.05, 0.05)

            # Save csv
            if save_csv is True:
                csv_stats.to_csv(
                    os.path.join(model_output, f"{model}_{x}_{csv_name}"), index=False
                )

        for stat in stats:
            # Convert to dataframe
            csv_item = csv_stats[stat].to_frame()

            # Rename column to point denisty
            csv_item.rename({stat: x}, axis=1, inplace=True)

            # Append dfs list
            if stat == "r2":
                dfs_r2.append(csv_item)
            if stat == "rmse":
                dfs_rmse.append(csv_item)

    # Concatenate dataframes
    df_r2 = pd.concat(dfs_r2, axis=1)
    df_rmse = pd.concat(dfs_rmse, axis=1)

    # Create Bar Chart for r2
    df_r2.plot.bar(width=0.9, edgecolor="black")
    plt.ylabel("r2")
    plt.grid(color="grey", linestyle="--", linewidth=0.1)
    plt.legend(title="Point Density")
    plt.tight_layout()

    # Save Figure
    if save_fig is True:
        plt.savefig(os.path.join(root_dir, f"{model}_r2.png"))
    plt.close()

    # Create Bar Chart for rmse
    df_rmse.plot.bar(width=0.9, edgecolor="black")
    plt.ylabel("rmse")
    plt.grid(color="grey", linestyle="--", linewidth=0.1)
    plt.legend(title="Point Density")
    plt.tight_layout()

    # Save Figure
    if save_fig is True:
        plt.savefig(os.path.join(root_dir, f"{model}_rmse.png"))
    plt.close()