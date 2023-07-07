""" common command line arguments and helper functions for GANs and U-Net """

import argparse
import pathlib

import torch
import torchvision

#import datasets.transforms
import datasets.transforms

def get_parser():
    """ returns common ArgumentParser for GANs and U-Net """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--crop",
        type=int,
        default=(256,),
        nargs="+",
        help="Size of crop. Can be a tuple of height and width",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=(256,),
        nargs="+",
        help="Resizing after cropping. Can be a tuple of height and width",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="number of epochs to train"
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument(
        "--suffix", help="suffix appended to the otuput directory", default=None
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="drc",
        choices=["drc", "gapfill"],
        help="Which dataset to use: DRC or cloud gap filling?",
    )
    parser.add_argument(
        "--dataroot", type=str, default="/workspace/data/gapfillfull", help="Path to dataset"
    )

    parser.add_argument(
        "--n_bands", type=int, default=4, help="Number of spectral bands"
    )
    parser.add_argument(
        "--time_steps", type=int, default=3, help="Number of time steps for gap filling dataset"
    )
    
    parser.add_argument(
        "--mask_position",
        type=int,
        default=[2],
        nargs="+",
        help="List of positions of mask in time steps - first time step = 1, last time step = input of --time_steps",
    )
    
    parser.add_argument(
        "--out_dir",
        type=pathlib.Path,
        default="/workspace/data/results",
        help="Where to store models, log, etc.",
    )

    return parser


def get_transforms(config):
    """ returns dataset transforms

    Parameters
    ----------
    config : dict
        configuration returned by args2dict

    Returns
    -------
    train and test transforms for the dataset

    """

    if config["dataset"]["name"] == "drc":
        train_transforms = torchvision.transforms.Compose(
            [
                datasets.transforms.ToTensor(),
                datasets.transforms.RandomCrop((192,192)),
                datasets.transforms.RandomHorizontalFlip(),
                datasets.transforms.RandomVerticalFlip(),
                datasets.transforms.OneHot(),
            ]
        )
        test_transforms = torchvision.transforms.Compose(
            [
                datasets.transforms.ToTensor(),
                datasets.transforms.CenterCrop((192,192)),
                datasets.transforms.OneHot(),
            ]
        )
        
    elif config["dataset"]["name"] == "gapfill":
        train_transforms = torchvision.transforms.Compose(
            [
                datasets.transforms.ToTensor(),
            ]
        )
        test_transforms = train_transforms
    else:
        raise RuntimeError("Invalid dataset. This should never happen")
    return train_transforms, test_transforms


def get_dataset(config, split, transforms):
    """ returns dataset

    Parameters
    ----------
    config : dict
        configuration returned by args2dict
    split : string
        use train or test split
    transforms
        train or test transforms returned by get_transforms

    Returns
    -------
    dataset class

    """

    name = config["dataset"]["name"]
    root = config["dataset"]["root"]

    if name == "gapfill":
        time_steps = config["dataset"]["time_steps"]
        mask_position = config["dataset"]["mask_position"]
        n_bands = config["dataset"]["n_bands"]
        return datasets.gapfill.GAPFILL(root, split, transforms, time_steps, mask_position, n_bands)
    if name == "drc":
        return datasets.drc.DRC(root, split, transforms)
    raise ValueError("Dataset must be nrw or dfc, but is {}".format(name))
