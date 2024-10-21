import os
import torch
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from data_utils.common import (
    IOStream,
    PointCloudsInPickle,
    _init_,
)
from data_utils.common import create_comp_csv, write_las
from data_utils.pts_augment import AugmentPointCloudsInPickle

def prepare_dataset(params):
    # Load datasets
    train_data_path = os.path.join(params["train_path"], str(params["num_points"]))
    train_pickle = params["train_pickle"]
    trainset = PointCloudsInPickle(train_data_path, train_pickle)
    trainset_idx = list(range(len(trainset)))
    rem = len(trainset_idx) % params["batch_size"]
    if rem <= 3:
        trainset_idx = trainset_idx[: len(trainset_idx) - rem]
        trainset = Subset(trainset, trainset_idx)
    if params["augment"] == True:
        for i in range(params["n_augs"]):
            aug_trainset = AugmentPointCloudsInPickle(
                train_data_path,
                train_pickle,
            )

            trainset = torch.utils.data.ConcatDataset([trainset, aug_trainset])
    train_loader = DataLoader(
        trainset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
        sampler=None,
        collate_fn=None,
        drop_last=True,
        pin_memory=True,
    )

    test_data_path = os.path.join(params["test_path"], str(params["num_points"]))
    test_pickle = params["test_pickle"]
    testset = PointCloudsInPickle(test_data_path, test_pickle)
    test_loader = DataLoader(
        testset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
        sampler=None,
        collate_fn=None,
        drop_last=True,
        pin_memory=True,
    )

    return train_loader, test_loader