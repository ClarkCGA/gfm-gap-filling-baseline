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

N_CHANNELS = {"img": 4} # CHANGE to however many channels are in the dataset

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

    Output of __getitem__ method:
    ----------
    A dictionary of tifs for a given sample with the following key tensor pairs for each given time step t:
    "truth{t}" : A tensor of the ground truth, unmasked, cloud free satellite imagery
    "cloud{t}" : A binary cloud mask tensor for each time step that is given in command line arguments
    "masked{t}" : A tensor of the ground truth with cloudy areas set to 0 according to the matching cloud mask
    "unmasked{t} : A tensor of the ground truth with non-cloudy areas set to 0 according to the matching cloud mask
    """

    splits = ["train", "test"]

    # 6/6: This will need to be rewritten when the structure of the data becomes clearer, but the output above is the important part - DWG
    def __init__(self, root, split="train", transforms=None, time_steps=3, mask_position=[1]):
        super().__init__(pathlib.Path(root), transforms=transforms)
        verify_str_arg(split, "split", self.splits)
        self.split = split
        self.time_steps = time_steps
        self.mask_position = mask_position
        self.timesteps = [f"truth{n}" for n in range(1, self.time_steps+1)]
        self.tif_paths = {timestep: self._get_tif_paths(timestep) for timestep in self.timesteps}
        self.cloud_paths = self._get_cloud_paths

    # Create list of all paths to a given timestep
    def _get_tif_paths(self, timestep):
        csv = pd.read_csv(self.root.joinpath("gapfillcatalog.csv"))
        if self.split == "test":
            catalog = csv[csv["usage"] == "validate"]
        else:
            catalog = csv[csv["usage"] == "train"]
        itemlist = sorted(catalog[f'dir_{timestep}'].tolist())
        pathlist = [self.root.joinpath(item) for item in itemlist]
        return pathlist
    
    # Create list of all paths to clouds
    def _get_cloud_paths(self):
        catalog = pd.read_csv(self.root.joinpath("cloudcatalog.csv"))
        itemlist = sorted(catalog['dir'].tolist())
        pathlist = [self.root.joinpath(item) for item in itemlist]
        return pathlist
    
    def __len__(self):
        return len(self.tif_paths["truth1"])

    def __getitem__(self, index):
        def read_tif_as_np_array(path):
            with rasterio.open(path) as src:
                return src.read()

        # Read all ground truths of the same spatial extent into the sample dictionary, excluding cloud scenes
        sample = {
            timestep: read_tif_as_np_array(self.tif_paths[timestep][index]) for timestep in self.timesteps
        }
        
        # Extract desired bands and rearrange to H,W,C for each time step
        for n in range(1, self.time_steps+1):
            sample[f"t{n}"] = self.extract_rgb_from_tif(sample.pop(f"t{n}"))

        # Read in randomly selected cloudy scene for each mask position
        for n in range(len(self.mask_position)):
            sample[f"cloud{self.mask_position[n]}"] = read_tif_as_np_array(self.cloud_paths[np.random.randint(0,len(self.cloud_paths)-1)])

        # Mask imagery at specified mask positions with randomly selected cloud scenes
        for n in range(len(self.mask_position)):
            sample[f"masked{self.mask_position[n]}"] = self.cloudmask(sample.pop(f"cloud{self.mask_position[n]}"), sample[f"t{self.mask_position[n]}"])

        # Unmask imagery at specified mask positions with randomly selected cloud scenes, to be used in comparing generated outputs to ground truth in training
        for n in range(len(self.mask_position)):
            sample[f"unmasked{self.mask_position[n]}"] = self.cloudmask(sample.pop(f"cloud{self.mask_position[n]}"), sample[f"t{self.mask_position[n]}"])

        # Perform any specified transforms on sample
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
    def cloudunmask(cloudtif, tif):
        unmasked_tif = np.copy(tif)  # Create a copy of the satellite imagery
        # Multiply satellite imagery by cloud mask - this sets non-cloudy pixels to 0
        unmasked_tif *= cloudtif
        return unmasked_tif
    
    @staticmethod
    def extract_rgb_from_tif(tif):
        # extracts desired bands from tif and arrage to H, W, C
        bands = np.empty((*tif.shape[1:], 3), dtype=tif.dtype)
        # 6/6: Rearrange according to order of bands
        bands[:, :, 0] = tif[3]
        bands[:, :, 1] = tif[2]
        bands[:, :, 2] = tif[1]
        bands[:, :, 3] = tif[0]
        bands = np.clip(bands, 0, 3500) # Clip to whatever the spectral range of the imagery is
        bands = (bands / np.max(bands)) # Normalize within range 0,1
        return rgb.astype(np.float32)

