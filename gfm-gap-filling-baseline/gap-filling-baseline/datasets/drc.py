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

classes = [
    "Non-Crop",
    "Crop",
    "Edge"
]

lcov_cmap = matplotlib.colors.ListedColormap(
    [
        "#009900",  # Non-Crop
        "#c6b044",  # Crop
        "#ffffff", # Edge
    ]
)
lcov_norm = matplotlib.colors.Normalize(vmin=1, vmax=2)

N_LABELS = 3  # +1 due to 0 having no label
N_CHANNELS = {"rgb": 3, "dem": 1, "seg": N_LABELS}


class DRC(VisionDataset):
    """ Dataset of DRC imagery, dems, and labelled crop fields.


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
    datatypes = ["rgb", "dem", "seg"]
    
    readers = {
        "sar": lambda path: Image.open(path).copy(),
        "rgb": lambda path: Image.open(path).convert("RGB"),
        "dem": lambda path: Image.open(path).copy(),
        "seg": lambda path: Image.open(path).convert("I;16"),
    }

    def __init__(self, root, split="train", transforms=None):
        super().__init__(pathlib.Path(root), transforms=transforms)
        verify_str_arg(split, "split", self.splits)
        self.split = split
        self.tif_paths = {dt: self._get_tif_paths(dt) for dt in self.datatypes}

    def _get_tif_paths(self, datatype):
        csv = pd.read_csv(self.root.joinpath("filtered_catalog_congo_cgan-123dept.csv"))
        if self.split == "test":
            catalog = csv[csv["usage"] == "validate"]
        else:
            catalog = csv[csv["usage"] == "train"]
        if datatype == "rgb":
            rgblist = sorted(catalog['dir_os'].tolist())
            rgbpath = [self.root.joinpath(item) for item in rgblist]
            return rgbpath
        elif datatype == "seg":
            seglist = sorted(catalog['dir_label'].tolist())
            segpath = [self.root.joinpath(item) for item in seglist]
            return segpath
        elif datatype == "dem":
            demlist = sorted(catalog['dir_dem'].tolist())
            dempath = [self.root.joinpath(item) for item in demlist]
            return dempath

    def __len__(self):
        return len(self.tif_paths["rgb"])

    def __getitem__(self, index):
        def read_tif_as_np_array(path):
            with rasterio.open(path) as src:
                return src.read()

        sample = {
            dt: read_tif_as_np_array(self.tif_paths[dt][index]) for dt in self.datatypes
        }
        
        # Also move channels to last dimension, which is expected by pytorch's to_tensor
        sample["rgb"] = self.extract_rgb_from_tif(sample.pop("rgb"))
        sample["seg"] = sample.pop("seg").transpose((1, 2, 0)).astype(np.float32)
        sample["dem"] = self.dem_norm(sample.pop("dem").transpose((1, 2, 0))).astype(np.float32)

        # Add the random noise vector to DEM:
        # sample["dem"] = self.dem_norm_rand(sample.pop("dem").transpose((1, 2, 0))).astype(np.float32)

        # Replace DEM with a random noise vector with gaussian dist:
        # sample["dem"] = self.dem_rand(sample.pop("dem").transpose((1, 2, 0)).astype(np.float32))


        if self.transforms:
            sample = self.transforms(sample)
        return sample

    @staticmethod
    def dem_norm(arr):
        # normalizes DEM to the interval [0, 1] using the min and max values of the entire dataset
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        return arr

    @staticmethod
    def dem_norm_rand(arr):
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        noise = np.random.normal(loc=0, scale=.1, size=arr.shape)
        noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
        arr = arr + noise
        return arr

    @staticmethod
    def dem_norm_all(arr):
        # normalizes DEM to the interval [0, 1] using the min and max values of the entire dataset and adds gaussian noise
        arr = (arr - 292) / (854 - 292)
        return arr
    
    @staticmethod
    def dem_rand(arr):
        # replaces dem with random noise vector
        arr = np.random.normal(loc=.5, scale=.1, size=arr.shape)
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        return arr

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

    @staticmethod
    def seg2rgb(segm):
        """ converts segmentation map to a plotable RGB image """
        return lcov_cmap(lcov_norm(segm))[:, :, :3]

    @staticmethod
    def depth2rgb(depth):
        """ converts DEM to a plotable RGB image """
        # depth -= np.min(depth)
        # depth /= np.max(depth)
        return matplotlib.cm.viridis(depth)[:, :, :3]