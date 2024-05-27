import logging

import torch
import torchvision
import tqdm
import numpy as np
import loss
from PIL import Image


logger = logging.getLogger(__name__)

class Visualizer:
    """
    Class for visualizing true color composites from a set of trained model weights

    Attributes:
        g_net (torch.nn.Model): a pytorch model used to generate missing pixels.
        d_net (torch.nn.Model): a pytorch model for determining if an input is true or generated.
        n_bands (int): the number of spectral bands in the model input.
        time_steps (int): the number of time steps in the model input.
        local_rank (int): the gpu which the visualization will use.
        out_dir (pathlib.Path): the directory in which to save files related .
        g_optim (torch.optim.Adam): a pytorch optimizer used for training the generator. 
        d_optim (torch.optim.Adam): a pytorch optimizer used for training the discriminator.
        
    Methods:
        visualize_tcc(self, idx, sample, dest_fake, disc_real, disc_fake, cloudmask): creates true color composite visualizations.
        d_one_step(self, idx, sample): one step of training the discriminator, with the option to create a true color composite.
        visualize(self, val_dataloader): the visualization pass.
    """
    def __init__(
        self, g_net, d_net, n_bands, time_steps, local_rank=0, out_dir=None  
    ):
        
        self.local_rank = local_rank
        self.rank = 0
        self.d_net = d_net
        self.g_net = g_net

        self.n_bands = n_bands
        self.time_steps = time_steps

        self.out_dir = out_dir
        self.vis_dir = out_dir / "visualizations"
        
    def visualize_tcc_no_disc(self, idx, sample, dest_fake, cloudmask):
        """ 
        Generate and save visualizations of inputs and outputs to the model as true color composites.
        This function will only create visualizations for the first tensor in the batch.
        This function creates visualizations of input, predicted, and ground truth images at all time steps.
        The resulting images are saved to the specified visualization path.

        Args:
            n_epoch (int): Current epoch number.
            idx (int): Index of the sample in the dataset.
            sample (dict): the model input in dictionary form.
            dest_fake (torch.tensor): a torch tensor representing the generator output of a reconstructed image.
            cloudmask (list of torch.Tensor): a list of torch tensors representing the rescaled cloud masks used in masking the discriminator output
        """
        # initialize empty lists for each category of visualization
        masked = []
        generated = []
        unmasked = []

        # visualize input images and masks for each time step
        for t in range(1, self.time_steps+1):
            masked_img = sample["masked"][0,(t-1)*self.n_bands:(t-1)*self.n_bands+3,:,:].clone().flip(0) * 3 
            cloud = sample["cloud"][0,(t-1)*self.n_bands:(t-1)*self.n_bands+3,:,:].clone()
            cloud_masked = torch.where(cloud == 1, cloud, masked_img)
            cloud_masked = torch.nn.functional.pad(cloud_masked, (2,2,2,2), value=0)
            masked.append(cloud_masked)
        
        # visualize generated model reconstructions for each time step
        for t in range(1, self.time_steps+1):
            gen_img = dest_fake[0,(t-1)*self.n_bands:(t-1)*self.n_bands+3,:,:].clone().flip(0) * 3
            masked_img = sample["masked"][0,(t-1)*self.n_bands:(t-1)*self.n_bands+3,:,:].clone().flip(0) * 3
            cloud = sample["cloud"][0,(t-1)*self.n_bands:(t-1)*self.n_bands+3,:,:].clone()
            gen_cloud_masked = gen_img * cloud
            gen_reconstruction = gen_cloud_masked + masked_img
            gen_result = torch.nn.functional.pad(gen_reconstruction, (2,2,2,2), value=0)
            generated.append(gen_result)

        # visualize ground truth for each time step
        for t in range(1, self.time_steps+1):
            unmasked_img = sample["unmasked"][0,(t-1)*self.n_bands:(t-1)*self.n_bands+3,:,:].clone().flip(0) * 3
            masked_img = sample["masked"][0,(t-1)*self.n_bands:(t-1)*self.n_bands+3,:,:].clone().flip(0) * 3
            unmasked_img += masked_img
            unmasked_img = torch.nn.functional.pad(unmasked_img, (2,2,2,2), value=0)
            unmasked.append(unmasked_img)

        # concatenate tensors vertically
        masked = torch.cat(masked, dim=1)
        generated = torch.cat(generated, dim=1)
        unmasked = torch.cat(unmasked, dim=1)

        # concatenate all tensors into 3 x 3 grid and save to the out directory
        torchvision.utils.save_image(
            torch.cat([masked]+[generated]+[unmasked], dim=2), self.vis_dir/ "idx{:04}_gen.jpg".format(idx),
        )

    def visualize_tcc(self, idx, sample, dest_fake, disc_real, disc_fake, cloudmask):
        """ 
        Generate and save visualizations of inputs and outputs to the model as true color composites.
        This function will only create visualizations for the first tensor in the batch.
        This function creates visualizations of input, predicted, and ground truth images at all time steps.
        The resulting images are saved to the specified visualization path.

        Args:
            idx (int): Index of the sample in the dataset.
            sample (dict): the model input in dictionary form.
            dest_fake (torch.tensor): a torch tensor representing the generator output of a reconstructed image.
            disc_real (list of torch.Tensor): a torch tensor representing the discriminator output for a ground truth image
            disc_fake (list of torch.tensor): a list of torch tensors representing the discriminator output for a reconstructed image
            cloudmask (list of torch.Tensor): a list of torch tensors representing the rescaled cloud masks used in masking the discriminator output
        """
        cloudmask_img = [tensor[0:1,6:7,:,:].clone().detach() for tensor in cloudmask] # get rescaled cloudmask images as a list of tensors
        cloudmask_rescale = [torch.nn.functional.interpolate(tensor, size=(224,224), mode='nearest')[0,:,:,:] for tensor in cloudmask_img] # rescale the image back to 224x224
        cloudmask_pad = [torch.nn.functional.pad(tensor, (2,2,2,2), value=1) for tensor in cloudmask_rescale] # add 2 pixel wide black border to image
        cloudmask_pad = [tensor.expand(3,228,228) for tensor in cloudmask_pad] # expand to 3 channels for visualizing as RGB

        # create a red tensor which will be used to denote areas of the discriminator output in non-cloudy locations
        red_tensor = torch.zeros_like(cloudmask_pad[0])
        red_tensor[0] = 1
        
        # get a list of tensors each representing the output of the patch discriminator at a different scale
        disc_real_img = [tensor[0:1,:,:,:].clone().detach() for tensor in disc_real]
        # upscale to 224 x 224
        disc_real_rescale = [torch.nn.functional.interpolate(tensor, size=(224,224), mode='nearest')[0,:,:,:] for tensor in disc_real_img]
        # add a black border around each tensor
        disc_real_pad = [torch.nn.functional.pad(tensor, (2,2,2,2), value=0) for tensor in disc_real_rescale]
        # expand to three channels for visualization
        disc_real_pad = [tensor.expand(3,228,228) for tensor in disc_real_pad]
        # set non-cloudy pixels to 0
        disc_real_pad = [tensor1 * tensor2 for tensor1, tensor2 in zip(disc_real_pad, cloudmask_pad)]
        # make non-cloudy pixels red
        disc_real_pad = [
            torch.where(cloud_tensor == 0, red_tensor, disc_tensor) for disc_tensor, cloud_tensor in zip(disc_real_pad, cloudmask_pad)
        ]

        # perform the same steps as above for outputs of discriminator from inputting the generated tensor
        disc_fake_img = [tensor[0:1,:,:,:].clone().detach() for tensor in disc_fake]
        disc_fake_rescale = [torch.nn.functional.interpolate(tensor, size=(224,224), mode='nearest')[0,:,:,:] for tensor in disc_fake_img]
        disc_fake_pad = [torch.nn.functional.pad(tensor, (2,2,2,2), value=0) for tensor in disc_fake_rescale]
        disc_fake_pad = [tensor.expand(3,228,228) for tensor in disc_fake_pad]
        disc_fake_pad = [tensor1 * tensor2 for tensor1, tensor2 in zip(disc_fake_pad, cloudmask_pad)]
        disc_fake_pad = [
            torch.where(cloud_tensor == 0, red_tensor, disc_tensor) for disc_tensor, cloud_tensor in zip(disc_fake_pad, cloudmask_pad)
        ]
        
        # initialize empty lists for each category of visualization
        masked = []
        generated = []
        unmasked = []

        # visualize input images and masks for each time step
        for t in range(1, self.time_steps+1):
            masked_img = sample["masked"][0,(t-1)*self.n_bands:(t-1)*self.n_bands+3,:,:].clone().flip(0) * 3 
            cloud = sample["cloud"][0,(t-1)*self.n_bands:(t-1)*self.n_bands+3,:,:].clone()
            cloud_masked = torch.where(cloud == 1, cloud, masked_img)
            cloud_masked = torch.nn.functional.pad(cloud_masked, (2,2,2,2), value=0)
            masked.append(cloud_masked)
        
        # visualize generated model reconstructions for each time step
        for t in range(1, self.time_steps+1):
            gen_img = dest_fake[0,(t-1)*self.n_bands:(t-1)*self.n_bands+3,:,:].clone().flip(0) * 3
            masked_img = sample["masked"][0,(t-1)*self.n_bands:(t-1)*self.n_bands+3,:,:].clone().flip(0) * 3
            cloud = sample["cloud"][0,(t-1)*self.n_bands:(t-1)*self.n_bands+3,:,:].clone()
            gen_cloud_masked = gen_img * cloud
            gen_reconstruction = gen_cloud_masked + masked_img
            gen_result = torch.nn.functional.pad(gen_reconstruction, (2,2,2,2), value=0)
            generated.append(gen_result)

        # visualize ground truth for each time step
        for t in range(1, self.time_steps+1):
            unmasked_img = sample["unmasked"][0,(t-1)*self.n_bands:(t-1)*self.n_bands+3,:,:].clone().flip(0) * 3
            masked_img = sample["masked"][0,(t-1)*self.n_bands:(t-1)*self.n_bands+3,:,:].clone().flip(0) * 3
            unmasked_img += masked_img
            unmasked_img = torch.nn.functional.pad(unmasked_img, (2,2,2,2), value=0)
            unmasked.append(unmasked_img)

        # replace first and last time steps for generated and ground truth with respective discriminator outputs
        generated[0]=disc_fake_pad[0]
        generated[2]=disc_fake_pad[1]
        unmasked[0]=disc_real_pad[0]
        unmasked[2]=disc_real_pad[1]

        # concatenate tensors vertically
        masked = torch.cat(masked, dim=1)
        generated = torch.cat(generated, dim=1)
        unmasked = torch.cat(unmasked, dim=1)

        # concatenate all tensors into 3 x 3 grid and save to the out directory
        torchvision.utils.save_image(
            torch.cat([masked]+[generated]+[unmasked], dim=2), self.out_dir / "images" / "idx{:04}_gen.jpg".format(idx),
        )

    def d_one_step(self, idx, sample):

        g_input = sample["masked"] # Generator input is the masked ground truth

        dest_fake = self.g_net(g_input).detach() # dest_fake is the output of the generator with a shape of [n_bands * n_timesteps, H, W]
        gen_unmasked = dest_fake * sample["cloud"] # gen_unmasked has a value of 0 at all non-cloud pixels
        gen_reconstruction = gen_unmasked + sample["masked"] # gen_reconstruction has generated values at masked pixels

        ground_truth = sample["masked"] + sample["unmasked"] # ground truth is all unmasked inputs and all masked inputs added together

        disc_real = self.d_net(g_input, ground_truth).final # returns a list of rescaled discriminator output tensors for the ground truth
        disc_fake = self.d_net(g_input, gen_reconstruction).final # returns a list of rescaled discrimnator output tensors for the generated reconstruction
        
        # Create a list of downscaled cloud masks to weight discriminator outputs, where any patch with a cloudy pixel counts as a cloud mask
        cloudmask = [torch.nn.functional.max_pool2d(sample["cloud"], 2 ** (3 + scale)) for scale in range(len(disc_real))]

        # run the visualize_tcc method
        self.visualize_tcc_no_disc(idx, sample, dest_fake, cloudmask)

    def visualize(self, val_dataloader):
        device = torch.device(f"cuda:{self.local_rank}")
        validation_pbar = tqdm.tqdm(
            range(len(val_dataloader)), colour="red", desc="Visualizations", leave=True
        )
        
        for idx, sample in enumerate(val_dataloader):
            if idx % 5 == 0:
                sample = {k: v.to(device) for k, v in sample.items()}
            
                with torch.no_grad():
                # Run the discriminator forward pass only for every 5th batch
                    self.d_one_step(idx, sample)
            validation_pbar.update(1)
                    
        return None
