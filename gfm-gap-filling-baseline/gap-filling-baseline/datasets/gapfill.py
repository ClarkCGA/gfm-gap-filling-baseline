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

N_CHANNELS = {"img": 6} # CHANGE to however many channels are in the dataset

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
    def __init__(self, root, split="train", transforms=None, time_steps=3, mask_position=[1], n_bands = 6, cloud_range = (0,1), training_length=8000, normalize=False,
                mean=[495.7316,  814.1386,  924.5740, 2962.5623, 2640.8833, 1740.3031], std=[286.9569, 359.3304, 576.3471, 892.2656, 945.9432, 916.1625]):
        super().__init__(pathlib.Path(root), transforms=transforms)

        verify_str_arg(split, "split", self.splits)
        self.root_dir = pathlib.Path(root)
        self.image_dir = self.root_dir.joinpath("chips_filtered/")
        self.cloud_dir = self.root_dir.joinpath("cloud_mask/")
        self.split = split
        self.time_steps = time_steps
        self.mask_position = mask_position
        self.n_bands = n_bands
        self.normalize = normalize
        self.training_length = training_length
        if self.split == "train":
            self.cloud_range = cloud_range
        if self.split == "validate":
            self.cloud_range = [0.01,1.0]
        if self.split == "train":
            self.tif_paths = self._get_tif_paths()[:self.training_length]
        if self.split == "validate":
            self.tif_paths = self._get_tif_paths()
        self.cloud_paths, self.cloud_catalog = self._get_cloud_paths()
        self.n_cloudpaths = len(self.cloud_paths)
        self.mean = np.array(mean * 3)[:, np.newaxis, np.newaxis]  # corresponding mean per band for normalization purpose
        self.std = np.array(std * 3)[:, np.newaxis, np.newaxis]  # corresponding std per band for normalization purpose
    
    # Create list of all merged image chips
    def _get_tif_paths(self):
        csv = pd.read_csv(self.root_dir.joinpath("final_chip_tracker.csv"))
        catalog = csv.loc[(csv["usage"] == self.split) & (csv["bad_pct_max"] < 5) & (csv["na_count"] == 0)]
        itemlist = sorted(catalog["chip_id"].tolist())
        pathlist = [self.image_dir.joinpath(f"{item}_merged.tif") for item in itemlist]
        chipslist = list(self.image_dir.glob("*.tif"))
        truelist = sorted(list(set(pathlist) & set(chipslist)))
        return truelist
    
    # Create list of all paths to clouds
    def _get_cloud_paths(self):
        csv = pd.read_csv(self.root_dir.joinpath("fmask_tracker.csv"))
        catalog = csv.loc[(csv["usage"] == self.split) & (csv["cloud_pct"] <= self.cloud_range[1]) & (csv["cloud_pct"] >= self.cloud_range[0])]
        itemlist = sorted(catalog["fmask_name"].tolist())
        chipslist = list(self.cloud_dir.glob("*.tif"))
        pathlist = [self.cloud_dir.joinpath(f"{item}") for item in itemlist]
        truelist = sorted(list(set(pathlist) & set(chipslist)))
        return truelist, catalog
    
    def __len__(self):
        return len(self.tif_paths)

    def __getitem__(self, index):
        def read_tif_as_np_array(path):
            with rasterio.open(path) as src:
                    return src.read()

        # Read in merged tif as ground truth
        groundtruth = read_tif_as_np_array(self.tif_paths[index]) # Scale factor for given image - maybe make it a command line arg later for generalizing
        
        # Initialize empty cloud mask with same dimensions as ground truth
        cloudbrick = np.zeros_like(groundtruth)

        # For every specified mask position, read in a random cloud scene and add to the block of cloud masks
        if self.split == "train" :
            for p in self.mask_position:
                cloudscene = read_tif_as_np_array(self.cloud_paths[np.random.randint(0,self.n_cloudpaths-1)]) # Read in random cloud scene
                cloudbrick[(p-1)*self.n_bands:p*self.n_bands,:,:] = cloudscene
                del cloudscene
        else:
            for p in self.mask_position:
                cloudscene = read_tif_as_np_array(self.cloud_paths[index % self.n_cloudpaths]) # Read in cloud scene in order
                cloudbrick[(p-1)*self.n_bands:p*self.n_bands,:,:] = cloudscene
                del cloudscene

        sample = {}
        sample['masked'] = self.normalize_tif(self.cloudmask(cloudbrick, groundtruth)).transpose(1,2,0).astype(np.float32)
        sample['unmasked'] = self.normalize_tif(self.cloudunmask(cloudbrick, groundtruth)).transpose(1,2,0).astype(np.float32)
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


    def normalize_tif(self, tif):
        if self.normalize:
            norm_tif = np.where(tif == -9999, 0.0001,
                                    (tif - self.mean) / self.std)  # don't normalize on nodata
        else:
            norm_tif = tif * 0.0001  # if not normalize, just rescale
        return norm_tif
