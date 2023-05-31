""" The dataset of the IEEE GRSS data fusion contest """

import pathlib
import logging

import rasterio
import numpy as np
import matplotlib
import pandas as pd

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import verify_str_arg
import torchvision.transforms.functional as TF

logging.getLogger("rasterio").setLevel(logging.WARNING)

N_LABELS = 0
N_CHANNELS = {"img": 8} # CHANGE to however many channels are in the dataset

class GAPFILL(VisionDataset):
    """ Dataset of imagery and cloud scenes for gap filling.


    Parameters
    ----------
    root : string
        Root directory of dataset
    split : string, optional
        Image split to use, ``train`` or ``test``
    transforms : callable, optional
        A function/transform that takes input sample and returns a transformed version.
    """

    splits = ["train", "test"]
    timesteps = ["t1", "t2", "t3", "cloud"]

    def __init__(self, root, split="train", transforms=None):
        super().__init__(pathlib.Path(root), transforms=transforms)
        verify_str_arg(split, "split", self.splits)
        self.split = split
        self.tif_paths = {timestep: self._get_tif_paths(timestep) for timestep in self.timesteps}

    def _get_tif_paths(self, timestep):
        csv = pd.read_csv(self.root.joinpath("gapfillcatalog.csv"))
        if self.split == "test":
            catalog = csv[csv["usage"] == "validate"]
        else:
            catalog = csv[csv["usage"] == "train"]
        if timestep == "t1":
            t1list = sorted(catalog['dir_t1'].tolist())
            t1path = [self.root.joinpath(item) for item in t1list]
            return t1path
        elif timestep == "t2":
            t2list = sorted(catalog['dir_t2'].tolist())
            t2path = [self.root.joinpath(item) for item in t2list]
            return t2path
        elif timestep == "t3":
            t3list = sorted(catalog['dir_t3'].tolist())
            t3path = [self.root.joinpath(item) for item in t3list]
            return t3path
        elif timestep == "cloud":
            cloudlist = sorted(catalog['dir_cloud'].tolist())
            cloudpath = [self.root.joinpath(item) for item in cloudlist]
            return cloudpath

    def __len__(self):
        return len(self.tif_paths["t1"])

    def __getitem__(self, index):
        def read_tif_as_np_array(path):
            with rasterio.open(path) as src:
                return src.read()

        sample = {
            timestep: read_tif_as_np_array(self.tif_paths[timestep][index]) for timestep in self.timesteps if timestep is not "cloud"
        }
        
        sample["cloud"] = read_tif_as_np_array(self.tif_paths["cloud"][np.random.randint(0,len(self.tif_paths["cloud"]))])

        # Also move channels to last dimension, which is expected by pytorch's to_tensor
        sample["t1"] = self.extract_rgb_from_tif(sample.pop("t1"))
        sample["t2"] = self.extract_rgb_from_tif(sample.pop("t2"))
        sample["t3"] = self.extract_rgb_from_tif(sample.pop("t3"))
        sample["cloud"] = cloudmask(sample.pop("cloud"), sample["t2"])

        if self.transforms:
            sample = self.transforms(sample)
        return sample

    @staticmethod
    def cloudmask(cloudtif, tif):
        masked_tif = np.copy(tif)  # Create a copy of the satellite imagery
        # Multiply satellite imagery by inverse of cloud mask - this sets cloudy pixels to 0
        masked_tif *= (1 - cloudtif)
        return masked_tif
    
    @staticmethod
    def extract_rgb_from_tif(tif):
        # extracts RGB bands from tif
        rgb = np.empty((*tif.shape[1:], 3), dtype=tif.dtype)
        rgb[:, :, 0] = tif[2]
        rgb[:, :, 1] = tif[1]
        rgb[:, :, 2] = tif[0]
        rgb = np.clip(rgb, 0, 3500)
        rgb = (rgb / np.max(rgb))
        return rgb.astype(np.float32)

