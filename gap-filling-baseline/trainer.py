import logging

import torch
import torchvision
import tqdm
import numpy as np
import loss
from PIL import Image

logger = logging.getLogger(__name__)

# ToDo
# Implement and test different learning rates for generator and discriminator and
# do multiple discriminator steps per generator step. See [1] and [2]
#
# [1] Heusel et. al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", 2018
# [2] Zhang et. al., "Self-Attention Generative Adversarial Networks", 2019


class Trainer:
    def __init__(
        self, g_net, d_net, visualization, n_bands, time_steps, alpha=4, out_dir=None 
    ):
        
        self.rank = 0
        self.alpha = alpha
        self.d_net = d_net
        self.g_net = g_net

        self.visualization = visualization
        self.n_bands = n_bands
        self.time_steps = time_steps

        self.out_dir = out_dir

        self.g_optim = torch.optim.Adam(
            self.g_net.parameters(), lr=0.0001, betas=(0, 0.9)
        )
        self.d_optim = torch.optim.Adam(
            self.d_net.parameters(), lr=0.0004, betas=(0, 0.9)
        )

        self.g_loss = loss.HingeGenerator()
        self.d_loss = loss.HingeDiscriminator()

    def visualize_tcc(self, n_epoch, idx, sample, dest_fake):
        masked = []
        generated = []
        unmasked = []
        for t in range(1, self.time_steps+1):
            masked_img = sample["masked"][0,(t-1)*self.n_bands:t*self.n_bands-1,:,:].clone().flip(0) * 3
            cloud = sample["cloud"][0,(t-1)*self.n_bands:t*self.n_bands-1,:,:].clone()
            cloud_masked = torch.where(cloud == 1, cloud, masked_img)
            cloud_masked = torch.nn.functional.pad(cloud_masked, (2,2,2,2), value=0)
            masked.append(cloud_masked)
        for t in range(1, self.time_steps+1):
            gen_img = dest_fake[0,(t-1)*self.n_bands:t*self.n_bands-1,:,:].clone().flip(0) * 3
            masked_img = sample["masked"][0,(t-1)*self.n_bands:t*self.n_bands-1,:,:].clone().flip(0) * 3
            gen_composite = torch.where(gen_img != 0, gen_img, masked_img)
            gen_composite = torch.nn.functional.pad(gen_composite, (2,2,2,2), value=0)
            generated.append(gen_composite)
        for t in range(1, self.time_steps+1):
            unmasked_img = sample["unmasked"][0,(t-1)*self.n_bands:t*self.n_bands-1,:,:].clone().flip(0) * 3
            masked_img = sample["masked"][0,(t-1)*self.n_bands:t*self.n_bands-1,:,:].clone().flip(0) * 3
            unmasked_img += masked_img
            unmasked_img = torch.nn.functional.pad(unmasked_img, (2,2,2,2), value=0)
            unmasked.append(unmasked_img)
        masked = torch.cat(masked, dim=2)
        generated = torch.cat(generated, dim=2)
        unmasked = torch.cat(unmasked, dim=2)
        torchvision.utils.save_image(
            torch.cat([masked]+[generated]+[unmasked], dim=1), self.out_dir/ "epoch{:04}_idx{:04}.jpg".format(n_epoch, idx),
        )

    def g_one_step(self, sample):
        self.g_optim.zero_grad()

        g_input = [sample["masked"], sample["cloud"]] # Generator input is the masked ground truth

        dest_fake = self.g_net(g_input)
        gen_composite = sample["masked"] + dest_fake

        d_output_fake = self.d_net(g_input[0], gen_composite).final
        cloudmask = [torch.nn.functional.avg_pool2d(sample["cloud"], 2 ** (3 + scale)) for scale in range(len(d_output_fake))]

        loss_val = sum(self.g_loss(*output) for output in zip(d_output_fake, cloudmask))
        
        # In this implementation of MSE loss, the mean squared error is normalized by the number of masked values we are generating.
        # This ensures that the model is not rewarded for non-generated pixel values.
        mse_normalized = torch.nn.functional.mse_loss(dest_fake, sample["unmasked"]) 
        if torch.mean(sample["cloud"]) != 0:
            mse_normalized /= torch.mean(sample["cloud"])

        loss_val += self.alpha * mse_normalized

        loss_val.backward()
        self.g_optim.step()

        return loss_val, mse_normalized

    def d_one_step(self, n_epoch, idx, sample):
        self.d_optim.zero_grad()

        g_input = [sample["masked"], sample["cloud"]]

        dest_fake = self.g_net(g_input).detach()
        dest_real = sample["unmasked"] # We are comparing generated unmasked values to the unmasked ground truth.

        if idx  == 1 and self.visualization == "image":
            self.visualize_tcc(n_epoch, idx, sample, dest_fake)

        ground_truth = sample["masked"] + sample["unmasked"]
        gen_composite = sample["masked"] + dest_fake

        disc_real = self.d_net(g_input[0], ground_truth).final
        disc_fake = self.d_net(g_input[0], gen_composite).final
        cloudmask = [torch.nn.functional.avg_pool2d(sample["cloud"], 2 ** (3 + scale)) for scale in range(len(disc_real))]

        loss_val = sum(
            self.d_loss(*disc_out) for disc_out in zip(disc_real, disc_fake, cloudmask)
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
            running_mse = torch.tensor(0.0, requires_grad=False)

            for idx, sample in enumerate(dataloader):
                sample = {k: v.to(device) for k, v in sample.items()}
                g_loss, mse = self.g_one_step(sample)
                running_g_loss += g_loss.item()
                running_mse = mse.item()

                d_loss = self.d_one_step(n_epoch, idx, sample)
                running_d_loss += d_loss.item()

                if self.rank == 0:
                    logger.debug(
                        "batch idx {:3d}, g_loss:{:7.3f}, d_loss:{:7.3f}".format(
                            idx, g_loss.item(), d_loss.item()
                        )
                    )
                
            running_g_loss /= len(dataloader)
            running_d_loss /= len(dataloader)
            running_mse /= len(dataloader)

            info_str = "epoch {:3d}, g_loss:{:7.3f}, d_loss:{:7.3f}, mse:{:7.3f}".format(
                n_epoch, running_g_loss, running_d_loss, running_mse
            )

            pbar.update(1)
            pbar.set_description(info_str)
            if self.rank == 0:
                pbar.write(info_str)
                logger.info(info_str)

        return None
