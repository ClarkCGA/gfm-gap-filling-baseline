import pathlib
import logging

import rasterio
import numpy as np
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
    time_steps : int, optional
        The number of time steps in the dataset, defaults to 3.
    mask_position : list of int, optional
        The time steps to load cloud masks into, defaults to middle step.
    n_bands: int, optional
        The number of bands in each time step, defaults to 6 for hls data
    cloud_range: tuple of int, optional
        The range of cloud coverage for cloudmasks used in training, defaults to between 0.01 and 1.
    training_length : int, optional
        The number of chips to use when trianing. Defaults to 6231, cannot be greater than the number of chips in the training dataset.
    normalize : bool, optional
        If True, z-normalization is applied with the dataset mean and standard deviation. If False, scaling factor normalization is applied
        with a scaling factor of 0.0001.
    mean : list of floats, optional
        per-band means for dataset, defaults to per-band means calculated for hls dataset
    std : list of floats, optional
        per-band standard deviations for dataset, defaults to per-band standard deviations calculated for hls dataset

    Output of __getitem__ method:
    ----------
    A dictionary of tifs for a given spatial extent with the following key tensor pairs:
    "masked" : A tensor of the ground truth with cloudy areas set to 0 according to the matching cloud mask, time steps are stacked in C dimension
    "unmasked" : A tensor of the ground truth with non-cloudy areas set to 0 according to the matching cloud mask, time steps are stacked in C dimension
    "cloud" : A tensor in which cloudy pixels are set to 1 and non-cloudy pixels are set to 0

    """

    splits = ["train", "validate"]

    def __init__(self, 
                 root, 
                 split="train", 
                 transforms=None, 
                 time_steps=3, 
                 n_bands = 6, 
                 cloud_range = (0.01,1), 
                 training_length=6231, 
                 normalize=False,
                 mean=[495.7316,  814.1386,  924.5740, 2962.5623, 2640.8833, 1740.3031], 
                 std=[286.9569, 359.3304, 576.3471, 892.2656, 945.9432, 916.1625]):
        super().__init__(pathlib.Path(root), transforms=transforms)

        verify_str_arg(split, "split", self.splits)

        # get all directories needed for reading in chips
        self.root_dir = pathlib.Path(root)
        self.image_dir = self.root_dir.joinpath("chips_filtered/")
        self.cloud_dir = self.root_dir.joinpath("cloud_mask/")
        
        # set parameters
        self.split = split
        self.time_steps = time_steps
        self.mask_position = [[1],[2],[3],[1,2],[2,3],[1,3],[1,2,3]]
        self.n_bands = n_bands
        self.normalize = normalize
        self.training_length = training_length

        # ensure that validation cloud range is always the same across experiments
        if self.split == "train":
            self.cloud_range = cloud_range
        if self.split == "validate":
            self.cloud_range = [0.01,1.0]

        # get image tif paths, a catalog of used tifs, cloud paths, and a catalog of used cloud masks using appropriate methods
        self.tif_paths, self.tif_catalog = self._get_tif_paths()
        self.cloud_paths, self.cloud_catalog = self._get_cloud_paths()

        self.n_cloudpaths = len(self.cloud_paths)

        self.mean = np.array(mean * 3)[:, np.newaxis, np.newaxis]  # corresponding mean per band for normalization purpose
        self.std = np.array(std * 3)[:, np.newaxis, np.newaxis]  # corresponding std per band for normalization purpose
    
    def _get_tif_paths(self):
        """
    Retrieve paths to valid image data files and their corresponding metadata catalog.

    This method reads a CSV file containing chip metadata and filters it based on the split, bad pixel percentage,
    and NA count criteria. It then creates a subset of the catalog and extracts valid chip IDs.
    The method constructs paths to the valid image data files and sorts the catalog by chip ID.

    Returns:
        tuple: A tuple containing two elements:
            - truelist (list): A list of pathlib.Path objects representing paths to valid image data files.
            - sorted_catalog (pd.DataFrame): A pandas DataFrame containing sorted metadata of valid chips.

    Note:
        The CSV file should be named "final_chip_tracker.csv" and located within the root directory.
    """
        
        csv = pd.read_csv(self.root_dir.joinpath("final_chip_tracker.csv")) # access chip tracker

        # filter csv by split, bad_pct_max and na_count
        catalog = csv.loc[(csv["usage"] == self.split) & (csv["bad_pct_max"] < 5) & (csv["na_count"] == 0)]
        
        # ensure that validation set is always the same across experiments
        if self.split == "train":
            catalog_subset = catalog.sample(n=self.training_length)
        else:
            catalog_subset = catalog

        itemlist = sorted(catalog_subset["chip_id"].tolist())
        pathlist = [self.image_dir.joinpath(f"{item}_merged.tif") for item in itemlist]  # get paths for each item of filtered catalog
        chipslist = list(self.image_dir.glob("*.tif")) # get paths for each item in the image directory
        truelist = sorted(list(set(pathlist) & set(chipslist))) # get only paths from the catalog which represent valid paths in the directory
        sorted_catalog = catalog_subset.sort_values(by="chip_id", ascending=True) # ensure that the catalog is sorted identically to the path list
        
        return truelist, sorted_catalog
    
    # Create list of all paths to clouds
    def _get_cloud_paths(self):
        """
    Retrieve paths to valid cloud mask data files and their corresponding metadata catalog.

    This method reads a CSV file containing cloud mask metadata and filters it based on the split and
    cloud percentage range. It then creates a catalog subset and extracts
    valid cloud mask filenames. The method constructs paths to the valid cloud mask data files.

    Returns:
        tuple: A tuple containing two elements:
            - truelist (list): A list of pathlib.Path objects representing paths to valid cloud mask data files.
            - catalog (pd.DataFrame): A pandas DataFrame containing metadata of valid cloud mask files.

    Note:
        The CSV file should be named "fmask_tracker_balanced.csv" and located within the root directory.
    """
        csv = pd.read_csv(self.root_dir.joinpath("fmask_tracker_balanced.csv")) # access cloud tracker

        # filter csv by usage and cloud cover range defined when initializing the dataset
        catalog = csv.loc[(csv["usage"] == self.split) & (csv["cloud_pct"] <= self.cloud_range[1]) & (csv["cloud_pct"] >= self.cloud_range[0])]
        
        itemlist = sorted(catalog["fmask_name"].tolist())
        chipslist = list(self.cloud_dir.glob("*.tif")) # get paths for each item in the cloud directory
        pathlist = [self.cloud_dir.joinpath(f"{item}") for item in itemlist] # get paths for each item of filtered catalog
        truelist = sorted(list(set(pathlist) & set(chipslist))) # get only paths from the catalog which represent valid paths in the directory
        
        return truelist, catalog
    
    def __len__(self):
        return len(self.tif_paths)

    def __getitem__(self, index):
        """
    Retrieve a combined data sample containing ground truth and cloud mask information.

    This method reads and processes image and cloud mask data for a given index. It loads the merged
    tif file as ground truth and optionally normalizes it according to the parameters passed when initializing the dataset class.
    It then creates an empty cloud mask array and populates it with cloud scenes based on the mask position and dataset split.

    Args:
        index (int): Index of the data sample to retrieve.

    Returns:
        A dictionary of tifs for a given spatial extent with the following key tensor pairs:
        "masked" : A tensor of the ground truth with cloudy areas set to 0 according to the matching cloud mask, time steps are stacked in C dimension
        "unmasked" : A tensor of the ground truth with non-cloudy areas set to 0 according to the matching cloud mask, time steps are stacked in C dimension
        "cloud" : A tensor in which cloudy pixels are set to 1 and non-cloudy pixels are set to 0

    Note:
        The method assumes that cloud mask data paths and ground truth data paths have been pre-loaded using the `get_cloud_paths` and `get_tif_paths` methods.
        Additionally, the cloud mask data is read randomly from available paths during training and cyclically during validation.
    """
        def read_tif_as_np_array(path):
            with rasterio.open(path) as src:
                    return src.read()

        # read in merged tif as ground truth
        groundtruth = read_tif_as_np_array(self.tif_paths[index])
        
        # initialize empty cloud mask with same dimensions as ground truth
        cloudbrick = np.zeros_like(groundtruth)

        mask_position = self.mask_position[index % 7] # this loops through the possible combinations of mask position

        # for every specified mask position, read in a random cloud scene and add to the block of cloud masks
        if self.split == "train" :
            for p in mask_position:
                cloudscene = read_tif_as_np_array(self.cloud_paths[np.random.randint(0,self.n_cloudpaths-1)]) # read in random cloud scene
                cloudbrick[(p-1)*self.n_bands:p*self.n_bands,:,:] = cloudscene
                del cloudscene
        else:
            for p in mask_position:
                cloudscene = read_tif_as_np_array(self.cloud_paths[(index + (p-1)) % self.n_cloudpaths]) # read in cloud scene in order
                cloudbrick[(p-1)*self.n_bands:p*self.n_bands,:,:] = cloudscene
                del cloudscene

        # create empty dictionary for sample
        sample = {}

        sample['masked'] = self.normalize_tif(self.cloudmask(cloudbrick, groundtruth)).transpose(1,2,0).astype(np.float32) # (C, H, W)
        sample['unmasked'] = self.normalize_tif(self.cloudunmask(cloudbrick, groundtruth)).transpose(1,2,0).astype(np.float32) # (C, H, W)
        sample['cloud'] = cloudbrick.transpose(1,2,0).astype(np.float32) # (C, H, W) with C = 1

        # perform any specified transforms on sample
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
