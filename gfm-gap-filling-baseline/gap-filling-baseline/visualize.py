import logging

import torch
import torch.nn
import torch.distributed

import pandas as pd
import numpy as np
import yaml

import datasets.drc
import options.common
import options.gan
from visualizer import Visualizer
import pathlib


##################################
#                                #
# Parsing command line arguments #
#                                #
##################################

parser = options.gan.get_parser()
args = parser.parse_args()

OUT_DIR = args.out_dir / args.checkpoint_dir

g_net_checkpoint = args.out_dir / args.checkpoint_dir / "model_gnet_best.pt"
d_net_checkpoint = args.out_dir / args.checkpoint_dir / "model_dnet_best.pt"

###################################
#                                 #
# Checking command line arguments #
#                                 #
###################################

# Reproducibilty config https://pytorch.org/docs/stable/notes/randomness.html
if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.warning(
        "You have chosen to seed training. "
        "This will turn on the CUDNN deterministic setting, "
        "which can slow down your training considerably! "
        "You may see unexpected behavior when restarting "
        "from checkpoints."
    )

# Create CONFIG, a dictionary of parameters used to create the dataset and GAN models
CONFIG = options.gan.args2dict(args)

if not torch.cuda.is_available():
    raise RuntimeError("This scripts expects CUDA to be available")

device = torch.device("cuda:{}".format(args.local_rank))

# set device of this process. Otherwise apex.amp throws errors.
# see https://github.com/NVIDIA/apex/issues/319
torch.cuda.set_device(device)

#########################
#                       #
# Dataset configuration #
#                       #
#########################
train_transforms, test_transforms = options.common.get_transforms(CONFIG)

val_dataset = options.common.get_dataset(CONFIG, split="validate", transforms=train_transforms)

print(f"Number of validation images: {len(val_dataset)}")
print(f"Number of validation cloud masks: {val_dataset.n_cloudpaths}")


################################
#                              #
# Neural network configuration #
#                              #
################################

g_net = options.gan.get_generator(CONFIG).to(device)
d_net = options.gan.get_discriminator(CONFIG).to(device)

if args.checkpoint_dir != pathlib.Path(""):
    g_net_state_dict = torch.load(g_net_checkpoint)
    d_net_state_dict = torch.load(d_net_checkpoint)

    g_net.load_state_dict(g_net_state_dict)
    d_net.load_state_dict(d_net_state_dict)

############
#          #
# Training #
#          #
############

visualizer = Visualizer(
    g_net,
    d_net,
    args.n_bands,
    args.time_steps,
    args.local_rank,
    out_dir=OUT_DIR,
    
)

val_sampler = torch.utils.data.SequentialSampler(val_dataset)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    sampler=val_sampler)

visualizer.visualize(val_dataloader)

