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
        self, g_net, d_net, src=["dem", "seg"], dest="rgb", feat_loss=None, out_dir=None
    ):
        
        self.rank = 0

        self.src = src
        self.dest = dest

        self.d_net = d_net
        self.g_net = g_net

        self.out_dir = out_dir

        # parameters taken from SPADE https://github.com/NVlabs/SPADE/issues/50#issuecomment-494217696
        self.g_optim = torch.optim.Adam(
            self.g_net.parameters(), lr=0.00001, betas=(0, 0.9)
        )
        self.d_optim = torch.optim.Adam(
            self.d_net.parameters(), lr=0.00004, betas=(0, 0.9)
        )

        self.g_loss = loss.HingeGenerator()
        self.g_feat_lambda = feat_loss
        if feat_loss is not None:
            self.g_feat_loss = torch.nn.functional.l1_loss
            self.d_net.return_intermed = True
        self.d_loss = loss.HingeDiscriminator()

    def sample2gen_input(self, sample):
        return {src: sample[src] for src in self.src}

    def g_one_step(self, sample):
    
        self.g_optim.zero_grad()

        g_input = self.sample2gen_input(sample)

        dest_fake = self.g_net(g_input)
        d_output_fake = self.d_net(g_input, dest_fake)

        loss_val = sum(self.g_loss(o) for o in d_output_fake.final)

        if self.g_feat_lambda is not None:
            dest_real = sample[self.dest]
            d_output_real = self.d_net(g_input, dest_real)
            if not d_output_real.features:
                logger.error("Trying to compute feature loss on empty list")
                raise RuntimeError("Trying to compute feature loss on empty list")

            feat_loss = sum(
                self.g_feat_loss(fake, real)
                for real, fake in zip(
                    d_output_real.features, d_output_fake.features
                )
            )
            loss_val += self.g_feat_lambda * feat_loss

        loss_val.backward()
        self.g_optim.step()

        return loss_val

    def d_one_step(self, sample):
        self.d_optim.zero_grad()

        g_input = self.sample2gen_input(sample)
        # call detach to not compute gradients for generator
        dest_fake = self.g_net(g_input).detach()
        dest_real = sample[self.dest]

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

            # Initialize empty list to contain rgb array for gif creator
            rgb_gif_arrays = []

            for idx, sample in enumerate(dataloader):
                
                # Code block below feeds the first sample into the generator and outputs it to an rgb image in results - DG
                # It now also extracts the ground truth sample, the true label, and the true dem
               
                """
                if idx == 100:
                    g_input = self.sample2gen_input(sample)
                    fake_rgb = self.g_net(g_input)
                    fake_rgb = (fake_rgb[0,:3,:,:].flip(1).squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
                    fake_rgb = np.transpose(fake_rgb, (1, 2, 0))
                    result = Image.fromarray(fake_rgb, mode='RGB')
                    result.save(self.out_dir / 'generated_epoch_{}_batch_{}.jpg'.format(n_epoch, idx))

                    true_rgb = (sample["rgb"][0,:3,:,:].flip(1).squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
                    true_rgb = np.transpose(true_rgb, (1, 2, 0))
                    result = Image.fromarray(true_rgb, mode='RGB')
                    result.save(self.out_dir / 'truth_epoch_{}_batch_{}.jpg'.format(n_epoch, idx))

                    true_label = (sample["seg"][0,1,:,:].squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
                    true_label = np.flipud(true_label)
                    result = Image.fromarray(true_label)
                    result.save(self.out_dir / 'label_epoch_{}_batch_{}.jpg'.format(n_epoch, idx))

                    true_dem = (sample["dem"][0,:,:].squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
                    result = Image.fromarray(true_dem)
                    result.save(self.out_dir / 'dem_epoch_{}_batch_{}.jpg'.format(n_epoch, idx))
                """

                # Code block below creates rgb array out of the first item in each batch and appends it to the list of arrays to generate the gif - DG
                """
                g_input = self.sample2gen_input(sample)
                fake_rgb = self.g_net(g_input)
                fake_rgb = (fake_rgb[0,:3,:,:].flip(1).squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
                fake_rgb = np.transpose(fake_rgb, (1, 2, 0))
                rgb_gif_arrays.append(fake_rgb)
                """
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

            # Write gif to results
            """
            with imageio.get_writer((self.out_dir / 'generated_epoch_{}.gif'.format(n_epoch)), mode='I', duration=0.02) as writer:
                for array in rgb_gif_arrays:
                    writer.append_data(array)
            """

            pbar.update(1)
            pbar.set_description(info_str)
            if self.rank == 0:
                pbar.write(info_str)
                logger.info(info_str)

        return None
