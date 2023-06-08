import logging

import torch
import tqdm
import numpy as np
import loss
from PIL import Image
import imageio

logger = logging.getLogger(__name__)

# ToDo
# Implement and test different learning rates for generator and discriminator and
# do multiple discriminator steps per generator step. See [1] and [2]
#
# [1] Heusel et. al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", 2018
# [2] Zhang et. al., "Self-Attention Generative Adversarial Networks", 2019


class Trainer:
    def __init__(
        self, g_net, d_net, src=["dem", "seg"], dest="rgb", out_dir=None
    ):
        
        self.rank = 0

        self.src = src
        self.dest = dest

        self.d_net = d_net
        self.g_net = g_net

        self.out_dir = out_dir

        self.g_optim = torch.optim.Adam(
            self.g_net.parameters(), lr=0.00001, betas=(0, 0.9)
        )
        self.d_optim = torch.optim.Adam(
            self.d_net.parameters(), lr=0.00004, betas=(0, 0.9)
        )

        self.g_loss = loss.HingeGenerator()
        self.d_loss = loss.HingeDiscriminator()

    def sample2gen_input(self, sample):
        
        if self.src == ["img"]: # Create a generator input dictionary containing masked cloud scenes where they exist
            # Create empty dictionary
            gen_input = {}
            # Sorted list of time steps as integers with no duplicates
            time_steps = sorted(list(set([int(k[-1]) for k in sample.keys()])))
            # Add masked images to the gen input
            for t in time_steps:
                gen_input[f"masked{t}"] = sample[f"masked{t}"]
            
            return gen_input

        else:
            return {src: sample[src] for src in self.src}

    def sample2disc_input(self, sample):
        # Create empty dictionary
        disc_input = {}
            # Sorted list of time steps as integers with no duplicates
        time_steps = sorted(list(set([int(k[-1]) for k in sample.keys()])))
            # Add masked images to the disc input in order
        for t in time_steps:
            disc_input[f"masked{t}"] = sample[f"masked{t}"]
        return disc_input
    
    def mask_gen_output(self, sample, gen_output):
        # function to take the gen output and mask it by the cloud mask at that position in the sample, to compare with unmasked
        # Inputs:
        # sample - a dict, keys are names of tensors and values are the tensors themselves
        # gen_output - a tensor of shape B, H, W, C, with C = number of time steps * number of channels
        # Create masking tensor of identical shape to gen_output, with cloud masking locations equal to 1, by concatenating cloud masks
        mask = torch.cat([sample[key] for key in sample if "cloud" in key], 1)
        masked_gen_output = gen_output * mask
        return masked_gen_output

    def g_one_step(self, sample):
    
        self.g_optim.zero_grad()

        g_input = self.sample2gen_input(sample)

        dest_fake = self.g_net(g_input)
        d_output_fake = self.d_net(g_input, dest_fake) # Modify to accept mask_gen_output

        loss_val = sum(self.g_loss(o) for o in d_output_fake.final)

        loss_val.backward()
        self.g_optim.step()

        return loss_val

    def d_one_step(self, sample):
        self.d_optim.zero_grad()

        g_input = self.sample2gen_input(sample)
        # call detach to not compute gradients for generator
        dest_fake = self.g_net(g_input).detach() # Modify to mask using mask_gen_output
        dest_real = torch.cat([sample[key] for key in self.dest], 1) # Modify to just read in unmasked for the number of time steps

        disc_real = self.d_net(g_input, dest_real).final
        disc_fake = self.d_net(g_input, dest_fake).final

        loss_val = sum(
            self.d_loss(*disc_out) for disc_out in zip(disc_real, disc_fake)
        )

        loss_val.backward()
        self.d_optim.step()

        return loss_val

    def train(self, dataloader, n_epochs):
        pbar = tqdm.tqdm(total=n_epochs)
        device = torch.device("cuda:0")
        for n_epoch in range(1, n_epochs + 1):
            running_g_loss = torch.tensor(0.0, requires_grad=False)
            running_d_loss = torch.tensor(0.0, requires_grad=False)

            for idx, sample in enumerate(dataloader):

                sample = {k: v.to(device) for k, v in sample.items()}
                g_loss = self.g_one_step(sample)
                running_g_loss += g_loss.item()

                d_loss = self.d_one_step(sample)
                running_d_loss += d_loss.item()

                if self.rank == 0:
                    logger.debug(
                        "batch idx {:3d}, g_loss:{:7.3f}, d_loss:{:7.3f}".format(
                            idx, g_loss.item(), d_loss.item()
                        )
                    )
                
            running_g_loss /= len(dataloader)
            running_d_loss /= len(dataloader)

            info_str = "epoch {:3d}, g_loss:{:7.3f}, d_loss:{:7.3f}".format(
                n_epoch, running_g_loss, running_d_loss
            )

            pbar.update(1)
            pbar.set_description(info_str)
            if self.rank == 0:
                pbar.write(info_str)
                logger.info(info_str)

        return None
