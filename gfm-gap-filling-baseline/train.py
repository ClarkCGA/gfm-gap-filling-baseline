import logging

import torch
import torch.nn
import torch.distributed

import pandas as pd
import numpy as np
import yaml

import options.common
import options.gan
from trainer import Trainer
import pathlib


##################################
#                                #
# Parsing command line arguments #
#                                #
##################################

parser = options.gan.get_parser()
args = parser.parse_args()

OUT_DIR = args.out_dir / options.gan.args2str(args)
OUT_DIR.mkdir(exist_ok=True)
VIS_DIR = OUT_DIR / "visualizations"
VIS_DIR.mkdir(exist_ok=True)

# read in checkpoints for generator and discriminator if they exist
g_net_checkpoint = args.out_dir / args.checkpoint_dir / "model_gnet_best.pt"
d_net_checkpoint = args.out_dir / args.checkpoint_dir / "model_dnet_best.pt"

###########
#         #
# Logging #
#         #
###########

logging.basicConfig(
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    level=logging.INFO,
    filename=OUT_DIR / "log_training.txt",
)
logger = logging.getLogger()
logger.info("Saving logs, configs and models to %s", OUT_DIR)


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

with open(OUT_DIR / "config.yml", "w") as cfg_file:
    yaml.dump(CONFIG, cfg_file)

if not torch.cuda.is_available():
    raise RuntimeError("This scripts expects CUDA to be available")

# set the device of this process to the local rank specified in the command line arguments
device = torch.device("cuda:{}".format(args.local_rank))
torch.cuda.set_device(device)

#########################
#                       #
# Dataset configuration #
#                       #
#########################

# use the CONFIG as arguments to get transforms as specified in the command line and in the datast configuration script
train_transforms, test_transforms = options.common.get_transforms(CONFIG)

# use the get_dataset script to create the training dataset according to the CONFIG
train_dataset = options.common.get_dataset(CONFIG, split="train", transforms=train_transforms)
train_dataset.cloud_catalog.to_csv(OUT_DIR / "training_clouds.csv", index=False)
train_dataset.tif_catalog.to_csv(OUT_DIR / "training_tifs.csv", index=False)

# use the get_dataset script to create the validation dataset according to the CONFIG
val_dataset = options.common.get_dataset(CONFIG, split="validate", transforms=train_transforms)
val_dataset.cloud_catalog.to_csv(OUT_DIR / "validate_clouds.csv", index=False)
val_dataset.tif_catalog.to_csv(OUT_DIR / "validate_tifs.csv", index=False)

# print the length of all datasets
print(f"Number of training images: {len(train_dataset)}")
print(f"Number of training cloud masks: {train_dataset.n_cloudpaths}")
print(f"Number of validation images: {len(val_dataset)}")
print(f"Number of validation cloud masks: {val_dataset.n_cloudpaths}")

# log the information on the datasets
logger.info(train_dataset)
logger.info(val_dataset)

################################
#                              #
# Neural network configuration #
#                              #
################################

# use the CONFIG to cretae the generator and discriminator
g_net = options.gan.get_generator(CONFIG).to(device)
d_net = options.gan.get_discriminator(CONFIG).to(device)

# if a checkpoint is specified in the command line argument, we load in the checkpoint
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

# create an instance of the Trainer class with the arguments passed in the command line
trainer = Trainer(
    g_net,
    d_net,
    args.visualization,
    args.n_bands,
    args.time_steps,
    args.generator_lr,
    args.discriminator_lr,
    args.alpha,
    args.local_rank,
    out_dir=OUT_DIR,
    
)

# set up dataloaders for training and validation
train_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=True)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler=train_sampler)

# we use a sequential sampler for validation, so the dataset is presented identically for all experiments
val_sampler = torch.utils.data.SequentialSampler(val_dataset)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    sampler=val_sampler)

# train the model, which will save the best models for the generator and discriminator
trainer.train(train_dataloader, val_dataloader, args.epochs)
