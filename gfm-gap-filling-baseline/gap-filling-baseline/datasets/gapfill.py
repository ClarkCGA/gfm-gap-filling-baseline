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
    A dictionary of tifs for a given spatial extent with the following key tensor pairs:
    "masked" : A tensor of the ground truth with cloudy areas set to 0 according to the matching cloud mask, time steps are stacked in C dimension
    "unmasked" : A tensor of the ground truth with non-cloudy areas set to 0 according to the matching cloud mask, time steps are stacked in C dimension
    """

    splits = ["train", "validate"]

    # 6/6: This will need to be rewritten when the structure of the data becomes clearer, but the output above is the important part - DWG
    def __init__(self, root, split="train", transforms=None, time_steps=3, mask_position=[1], n_bands = 4):
        super().__init__(pathlib.Path(root), transforms=transforms)

        verify_str_arg(split, "split", self.splits)

        self.split = split
        self.time_steps = time_steps
        self.mask_position = mask_position
        self.n_bands = n_bands
        self.tif_paths = self._get_tif_paths()
        self.cloud_paths = self._get_cloud_paths()

        self.n_cloudpaths = len(self.cloud_paths)

    # Create list of all merged image chips
    def _get_tif_paths(self):
        csv = pd.read_csv(self.root.joinpath("chip_tracker.csv"))
        catalog = csv[csv["usage"] == self.split]
        itemlist = sorted(catalog['chip_id'].tolist())
        pathlist = [self.root.joinpath(f"chip_{item}_merged.tif") for item in itemlist]
        return pathlist
    
    # Create list of all paths to clouds
    def _get_cloud_paths(self):
        csv = pd.read_csv(self.root.joinpath("chip_tracker.csv"))
        catalog = csv[csv["usage"] == self.split]
        itemlist = sorted(catalog['chip_id'].tolist())
        pathlist = [self.root.joinpath(f"chip_{item}.mask.tif") for item in itemlist]
        return pathlist
    
    def __len__(self):
        return len(self.tif_paths)

    def __getitem__(self, index):
        def read_tif_as_np_array(path):
            with rasterio.open(path) as src:
                    return src.read()

        # Read in merged tif as ground truth
        groundtruth = read_tif_as_np_array(self.tif_paths[index]) / 10000 # Scale factor for given image - maybe make it a command line arg later for generalizing

        # Initialize empty cloud mask with same dimensions as ground truth
        # cloudbrick = np.zeros_like(groundtruth)

        # For every specified mask position, read in a random cloud scene and add to the block of cloud masks
        # for p in self.mask_position:
        #    cloudscene = read_tif_as_np_array(self.cloud_paths[np.random.randint(0,self.n_cloudpaths-1)]) # Read in random cloud scene
        #    cloudbrick[(p-1)*self.n_bands:p*self.n_bands,:,:] = cloudscene # Check if this works, the code should assign cloud scene to ALL these values in the 4 channels indexed.
        #    del cloudscene

        randbrick = np.random.choice([0,1], size=(1, 14, 14), p=[0.5, 0.5])
        cloudscene = randbrick.repeat(16, axis=1).repeat(16, axis=2).repeat(4, axis=0)
        cloudbrick = np.zeros_like(groundtruth)
        cloudbrick[4:8, :, :] = cloudscene

        sample = {}
        sample['masked'] = self.cloudmask(cloudbrick, groundtruth).transpose(1,2,0).astype(np.float32)
        sample['unmasked'] = self.cloudunmask(cloudbrick, groundtruth).transpose(1,2,0).astype(np.float32)
        sample['cloud'] = cloudbrick.transpose(1,2,0).astype(np.float32)

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

