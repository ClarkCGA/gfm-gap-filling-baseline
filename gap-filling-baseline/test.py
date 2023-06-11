import argparse
import pathlib

import torch
import torch.nn
import torchvision
import numpy as np
import tqdm
import yaml

import datasets.gapfill
import options.gan as options
from utils import unwrap_state_dict


###############################################
#                                             #
# Parsing and checking command line arguments #
#                                             #
###############################################

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("model", type=str)
args = parser.parse_args()

print("loading model {}".format(args.model))
# infer output directory from model path
OUT_DIR = pathlib.Path(args.model).absolute().parents[0]

# loading config
with open(OUT_DIR / "config.yml", "r") as stream:
    CONFIG = yaml.load(stream, Loader=yaml.FullLoader)
print("config: {}".format(CONFIG))

n_bands = CONFIG["dataset"]["n_bands"]
time_steps = CONFIG["dataset"]["time_steps"]

train_transforms, test_transforms = options.common.get_transforms(CONFIG)

dataset = options.common.get_dataset(CONFIG, split='validate', transforms=test_transforms)


###########
#         #
# Testing #
#         #
###########

if torch.cuda.device_count() >= 1:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = options.get_generator(CONFIG)
# remove distributed wrapping, i.e. module. from keynames
state_dict = torch.load(args.model)
model.load_state_dict(state_dict)
model.eval()
model.to(device)


############
#          #
# Plotting #
#          #
############

BATCH_SIZE = 1

test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

with torch.no_grad():
    for idx, sample in tqdm.tqdm(
        enumerate(test_dataloader), total=len(test_dataloader)
    ):
        sample = {k: v.to(device) for k, v in sample.items()}
        g_input = [sample["masked"], sample["cloud"]] # Generator input is the masked ground truth
        dest_fake = model(g_input)
        # Calculate normalized mse
        mse_normalized = torch.nn.functional.mse_loss(dest_fake, sample["unmasked"]) 
        if torch.mean(sample["cloud"]) != 0:
            mse_normalized /= torch.mean(sample["cloud"])

        masked = []
        generated = []
        unmasked = []
        for t in range(1, time_steps+1):
            masked_img = sample["masked"][0,(t-1)*n_bands:t*n_bands-1,:,:].clone() * 3
            cloud = sample["cloud"][0,(t-1)*n_bands:t*n_bands-1,:,:].clone()
            cloud_masked = torch.where(cloud == 1, cloud, masked_img)
            cloud_masked = torch.nn.functional.pad(cloud_masked, (2,2,2,2), value=0)
            masked.append(cloud_masked)
        for t in range(1, time_steps+1):
            gen_img = dest_fake[0,(t-1)*n_bands:t*n_bands-1,:,:].clone() * 3
            masked_img = sample["masked"][0,(t-1)*n_bands:t*n_bands-1,:,:].clone() * 3
            gen_composite = torch.where(gen_img != 0, gen_img, masked_img)
            gen_composite = torch.nn.functional.pad(gen_composite, (2,2,2,2), value=0)
            generated.append(gen_composite)
        for t in range(1, time_steps+1):
            unmasked_img = sample["unmasked"][0,(t-1)*n_bands:t*n_bands-1,:,:].clone() * 3
            masked_img = sample["masked"][0,(t-1)*n_bands:t*n_bands-1,:,:].clone() * 3
            unmasked_img += masked_img
            unmasked_img = torch.nn.functional.pad(unmasked_img, (2,2,2,2), value=0)
            unmasked.append(unmasked_img)
        masked = torch.cat(masked, dim=2)
        generated = torch.cat(generated, dim=2)
        unmasked = torch.cat(unmasked, dim=2)
        torchvision.utils.save_image(
            torch.cat([masked]+[generated]+[unmasked], dim=1), OUT_DIR / "test_{:04}_mse_{:.4f}.jpg".format(idx, mse_normalized),
        )

