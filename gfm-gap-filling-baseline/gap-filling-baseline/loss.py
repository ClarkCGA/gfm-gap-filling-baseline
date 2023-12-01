import torch
import torch.nn as nn


class HingeDiscriminator(nn.Module):
    """ Hinge loss for discriminator

    [1] Jae Hyun Lim, Jong Chul Ye, "Geometric GAN", 2017
    [2] Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida,
        "Spectral normalization for generative adversarial networks", 2018

    """

    def __init__(self):
        super().__init__()

    def forward(self, disc_real_output, disc_fake_output, cloudmask):
        """

        Args:
        disc_real_output: the discriminators output for a real sample
        disc_fake_output: the discriminators output for a fake sample

        """

        loss = -torch.mean(
            torch.min(disc_real_output * cloudmask - 1, torch.zeros_like(disc_real_output)) * cloudmask
        ) # Below 1 is incorrectly classified as fake. Penalize according to the mean of all incorrect classifications
        loss -= torch.mean(
            torch.min(-disc_fake_output * cloudmask - 1, torch.zeros_like(disc_fake_output)) * cloudmask
        ) # Below 1 is correctly classified as fake. Reward according to the mean of all correct classifications

        return loss


class HingeGenerator(nn.Module):
    """ Hinge loss for discriminator

    [1] Jae Hyun Lim, Jong Chul Ye, "Geometric GAN", 2017
    [2] Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida,
        "Spectral normalization for generative adversarial networks", 2018

    """

    def __init__(self):
        super().__init__()

    def forward(self, disc_fake_output, cloudmask):
        return -torch.mean(disc_fake_output * cloudmask)
        # Take the average classification of the discriminator as loss