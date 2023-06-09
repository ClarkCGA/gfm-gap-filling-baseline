""" command line arguments and helper functions for GANs """

import argparse
import datetime

import models.generator
import models.discriminator
import datasets
from . import common


def get_parser():
    """ returns ArgumentParser for GANs """

    # Get common ArgumentParser.
    parser = common.get_parser()

    # Add GAN specific arguments to the parser.

    parser.add_argument(
        "--local_rank", type=int, default=0, help="for distributed training"
    )
    parser.add_argument(
        "--n_sampling",
        default=4,
        type=int,
        help="number of upsampling/downsampling in the generator",
    )
    parser.add_argument(
        "--model_cap",
        default=64,
        type=int,
        choices=[16, 32, 48, 64],
        help="Model capacity, i.e. number of features.",
    )
    parser.add_argument(
        "--n_scales",
        default=2,
        type=int,
        help="Number of scales for multiscale discriminator",
    )
    parser.add_argument(
        "--visualization",
        default=None,
        type=str,
        choices=["image"],
        help="Visualization style during training loop.",
    )

    return parser


def args2str(args):
    """ converts arguments to string

    Parameters
    ----------
    args: arguments returned by parser

    Returns
    -------
    string of arguments

    """


    # training arguments
    train_str = "{args.dataset}_bs{args.batch_size}_ep{args.epochs}_cap{args.model_cap}".format(
        args=args
    )

    if args.seed:
        train_str += "_rs{args.seed}".format(args=args)

    datestr = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

    idstr = "_".join([train_str, datestr])
    if args.suffix:
        idstr = idstr + "_{}".format(args.suffix)
    return idstr


def args2dict(args):
    """ converts arguments to dict

    Parameters
    ----------
    args: arguments returned by parser

    Returns
    -------
    dict of arguments

    """

    # model_parameters
    model_args = ["model_cap", "n_sampling", "n_scales"]
    train_args = ["crop", "resize", "batch_size", "epochs", "visualization"]
    if args.seed:
        train_args.append("seed")
    model = {param: getattr(args, param) for param in model_args}
    train = {param: getattr(args, param) for param in train_args}
    data = {
        "name": args.dataset,
        "root": args.dataroot,
        "time_steps": args.time_steps,
        "mask_position": args.mask_position,
        "n_bands": args.n_bands
    }

    return {"model": model, "training": train, "dataset": data}


def get_generator(config):
    """ returns generator

    Parameters
    ----------
    config : dict
        configuration returned by args2dict

    Returns
    -------
    torch.nn.Model of generator

    """

    dset_class = getattr(datasets, config["dataset"]["name"])

    # Generator for gap filling task
    if config["dataset"]["name"]=="gapfill":
        # Set input channels of n_bands multiplied by number of time steps
        input_nc = config["dataset"]["n_bands"] * config["dataset"]["time_steps"]
        # Set output channels to equal input_nc
        output_nc = input_nc
        return models.generator.ResnetEncoderDecoder(
            input_nc,
            config["model"]["model_cap"],
            output_nc,
            n_downsample=config["model"]["n_sampling"],
        )
    # Maintain functionality for testing with DRC
    else:
        # Proposed conventional generator with SPADE norm layers everywhere
        n_labels = dset_class.N_LABELS
        output_nc = dset_class.N_CHANNELS[config["dataset"]["output"][0]]
        input_nc = sum(dset_class.N_CHANNELS[it] for it in config["dataset"]["input"] if it != "seg")
        return models.generator.SPADEResnetEncoderDecoder(
            input_nc,
            n_labels,
            config["model"]["model_cap"],
            output_nc,
            n_downsample=config["model"]["n_sampling"],
        )


def get_discriminator(config):
    """ returns discriminator

    Parameters
    ----------
    config : dict
        configuration returned by args2dict

    Returns
    -------
    torch.nn.Model of discriminator
    """
    if config["dataset"]["name"] == "drc":
        dset_class = getattr(datasets, config["dataset"]["name"])
    # generator conditioned on this input

        gen_input_nc = sum(
            dset_class.N_CHANNELS[it] for it in config["dataset"]["input"] if it != "seg"
        )
        if "seg" in config["dataset"]["input"]:
            gen_input_nc += dset_class.N_LABELS

        disc_input_nc = gen_input_nc + dset_class.N_CHANNELS[config["dataset"]["output"][0]]

    # Input number of time steps of conditions + number of time steps of output, multiplied by channels of the imagery.
    # As input_nc = output_nc, we can multiply input nc (time steps) by 2 and then multiply by n channels.
    if config["dataset"]["name"]=="gapfill":
        disc_input_nc = config["dataset"]["time_steps"] * config["dataset"]["n_bands"] * 2
    
    # Downsampling is done in the multiscale discriminator,
    # i.e., all discriminators are identically configures
    d_nets = [
        models.discriminator.PatchGAN(input_nc=disc_input_nc, init_nc=64)
        for _ in range(config["model"]["n_scales"])
    ]

    return models.discriminator.Multiscale(d_nets)
