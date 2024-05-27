import logging

import torch
import torchvision
import tqdm
import numpy as np
import loss
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.regression import MeanSquaredError
from torchmetrics.regression import MeanAbsoluteError


logger = logging.getLogger(__name__)

class Trainer:
    """
    Class for training of the cloud gap imputation CGAN model.

    Attributes:
        g_net (torch.nn.Model): a pytorch model used to generate missing pixels.
        d_net (torch.nn.Model): a pytorch model for determining if an input is true or generated.
        visualization (str): the type of visualization to produce.
        n_bands (int): the number of spectral bands in the model input.
        time_steps (int): the number of time steps in the model input.
        generator_lr (float): the learning rate of the generator.
        discriminator (float): the learning rate of the discriminator.
        alpha (float): hyperparameter defining weight given to hinge loss versus mean squared error loss.
        local_rank (int): the gpu which the training loop will use.
        out_dir (pathlib.Path): the directory in which to save files related to training.
        g_optim (torch.optim.Adam): a pytorch optimizer used for training the generator. 
        d_optim (torch.optim.Adam): a pytorch optimizer used for training the discriminator.
        
    Methods:
        visualize_tcc(self, n_epoch, idx, sample, dest_fake, disc_real, disc_fake, cloudmask): creates true color composite visualizations during training.
        g_one_step(self, sample, split): one step of the training the generator
        d_one_step(self, n_epoch, idx, sample, split): one step of training the discriminator, with the option to create a true color composite.
        train(self, train_dataloader, val_dataloader, n_epochs): the main training loop.
    """
    def __init__(
        self, g_net, d_net, visualization, n_bands, time_steps, generator_lr = 0.00001, discriminator_lr = 0.00004, alpha=4, local_rank=0, out_dir=None  
    ):
        
        # setting up training parameters from arguments
        self.local_rank = local_rank
        self.rank = 0
        self.alpha = alpha
        self.d_net = d_net
        self.g_net = g_net
        self.visualization = visualization
        self.n_bands = n_bands
        self.time_steps = time_steps
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.out_dir = out_dir
        self.vis_dir = out_dir / "visualizations"
    

        # setting up optimizers
        self.g_optim = torch.optim.Adam(
            self.g_net.parameters(), lr=self.generator_lr, betas=(0, 0.9)
        )
        self.d_optim = torch.optim.Adam(
            self.d_net.parameters(), lr=self.discriminator_lr, betas=(0, 0.9)
        )
        
        # setting device as local rank
        self.device = torch.device(f"cuda:{self.local_rank}")

        # send training metrics to device
        self.mean_squared_error = MeanSquaredError().to(self.device)
        self.mean_abs_error = MeanAbsoluteError().to(self.device)
        self.structural_similarity = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        
        # set up loss metrics
        self.g_loss = loss.HingeGenerator()
        self.d_loss = loss.HingeDiscriminator()

    def visualize_tcc(self, n_epoch, idx, sample, dest_fake, disc_real, disc_fake, cloudmask):
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
            torch.cat([masked]+[generated]+[unmasked], dim=2), self.vis_dir/ "epoch{:04}_idx{:04}_gen.jpg".format(n_epoch, idx),
        )

    def visualize_tcc_no_disc(self, n_epoch, idx, sample, dest_fake, cloudmask):
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
            torch.cat([masked]+[generated]+[unmasked], dim=2), self.vis_dir/ "epoch{:04}_idx{:04}_gen.jpg".format(n_epoch, idx),
        )

    def g_one_step(self, sample, split):
        self.g_optim.zero_grad()

        g_input = sample["masked"] # Generator input is the masked ground truth

        dest_fake = self.g_net(g_input) # dest_fake is the output of the generator with a shape of [n_bands * n_timesteps, H, W]
        gen_unmasked = dest_fake * sample["cloud"] # gen_unmasked has a value of 0 at all non-cloud pixels
        gen_reconstruction = gen_unmasked + sample["masked"] # gen_reconstruction has generated values at masked pixels
        d_output_fake = self.d_net(g_input, gen_reconstruction).final # d_output_fake is a list of tensors representing discriminator outputs at different scales
        cloud_mean = torch.mean(sample["cloud"])
        
        # Create a list of downscaled cloud masks to weight discriminator outputs, where any patch with a cloudy pixel counts as a cloud mask
        cloudmask = [torch.nn.functional.max_pool2d(sample["cloud"], 2 ** (3 + scale)) for scale in range(len(d_output_fake))]

        # get the hinge loss, only calculating for patches which have a cloudy pixel
        loss_val = sum(self.g_loss(*output) for output in zip(d_output_fake, cloudmask))
        
        # In this implementation of MSE loss, the mean squared error is normalized by the number of masked values we are generating.
        # This ensures that the model is not rewarded for non-generated pixel values.
        mse_normalized = torch.nn.functional.mse_loss(gen_unmasked, sample["unmasked"]) 
        mse_normalized /= torch.mean(sample["cloud"])
        
        # get the mean squared error normalized by the mean cloud mask for the center time step
        mse_score = self.mean_squared_error(gen_unmasked, sample["unmasked"])
        mse_score /= cloud_mean

        # get the mean absolute error normalized by the mean cloud mask for the center time step
        mae_score = self.mean_abs_error(gen_unmasked, sample["unmasked"])
        mae_score /= cloud_mean
        
        # get the ssim, do not normalize with the cloud mask.
        ssim = self.structural_similarity(gen_unmasked, sample["unmasked"])

        # combine hinge loss with mean squared error loss according to hyperparameter alpha
        loss_val += self.alpha * mse_normalized
        
        if split == "train":
            loss_val.backward()
            self.g_optim.step()

        return loss_val, mse_score, mae_score, ssim

    def d_one_step(self, n_epoch, idx, sample, split):
        self.d_optim.zero_grad()

        g_input = sample["masked"] # Generator input is the masked ground truth

        dest_fake = self.g_net(g_input).detach() # dest_fake is the output of the generator with a shape of [n_bands * n_timesteps, H, W]
        gen_unmasked = dest_fake * sample["cloud"] # gen_unmasked has a value of 0 at all non-cloud pixels
        gen_reconstruction = gen_unmasked + sample["masked"] # gen_reconstruction has generated values at masked pixels

        ground_truth = sample["masked"] + sample["unmasked"] # ground truth is all unmasked inputs and all masked inputs added together

        disc_real = self.d_net(g_input, ground_truth).final # returns a list of rescaled discriminator output tensors for the ground truth
        disc_fake = self.d_net(g_input, gen_reconstruction).final # returns a list of rescaled discrimnator output tensors for the generated reconstruction
        
        # Create a list of downscaled cloud masks to weight discriminator outputs, where any patch with a cloudy pixel counts as a cloud mask
        cloudmask = [torch.nn.functional.max_pool2d(sample["cloud"], 2 ** (3 + scale)) for scale in range(len(disc_real))]
        
        # get the discrimnator hinge loss, only calculating for patches which have a cloudy pixel
        loss_val = sum(
            self.d_loss(*disc_out) for disc_out in zip(disc_real, disc_fake, cloudmask)
        )
        
        # run the visualize_tcc method for the 5th batch in each loop
        if idx % 5 == 0 and split == "validate" and self.visualization == "image":
            self.visualize_tcc_no_disc(n_epoch, idx, sample, dest_fake, cloudmask)

        if split == "train":
            loss_val.backward()
            self.d_optim.step()

        return loss_val

    def train(self, train_dataloader, val_dataloader, n_epochs):
        pbar = tqdm.tqdm(total=n_epochs, desc="Overall Training", leave=True)
        device = torch.device(f"cuda:{self.local_rank}")
        best_loss = torch.tensor([100.0])

        for n_epoch in range(1, n_epochs + 1):
            training_pbar = tqdm.tqdm(
                range(len(train_dataloader)), colour="blue", desc="Training Epoch", leave=True
            )

            # set up running loss tensors
            running_g_loss = torch.tensor(0.0, requires_grad=False)
            running_d_loss = torch.tensor(0.0, requires_grad=False)
            running_mse = torch.tensor(0.0, requires_grad=False)
            running_mae = torch.tensor(0.0, requires_grad=False)
            running_ssim = torch.tensor(0.0, requires_grad=False)
            
            val_g_loss = torch.tensor(0.0, requires_grad=False)
            val_d_loss = torch.tensor(0.0, requires_grad=False)
            val_mse = torch.tensor(0.0, requires_grad=False)
            val_mae = torch.tensor(0.0, requires_grad=False)
            val_ssim = torch.tensor(0.0, requires_grad=False)

            for idx, sample in enumerate(train_dataloader):
                
                # run the generator forward pass
                sample = {k: v.to(device) for k, v in sample.items()}
                g_loss, mse, mae, ssim = self.g_one_step(sample, "train")
                running_g_loss += g_loss.item()
                running_mse += mse.item()
                running_mae += mae.item()
                running_ssim += ssim.item()

                # run the discrimnator forward pass
                d_loss = self.d_one_step(n_epoch, idx, sample, "train")
                running_d_loss += d_loss.item()

                training_pbar.update(1)
                
                if self.rank == 0:
                    logger.debug(
                        "train batch idx {:3d}, g_loss:{:7.3f}, d_loss:{:7.3f}".format(
                            idx, g_loss.item(), d_loss.item()
                        )
                    )
  
            running_g_loss /= len(train_dataloader)
            running_d_loss /= len(train_dataloader)
            running_mse /= len(train_dataloader)
            running_mae /= len(train_dataloader)
            running_ssim /= len(train_dataloader)

            # log training statistics
            training_info_str = "epoch {:3d}, train_g_loss:{:7.3f}, train_d_loss:{:7.3f}, train_mse:{:7.8f}, train_mae:{:7.4f}, train_ssim:{:7.8f}".format(
                n_epoch, running_g_loss, running_d_loss, running_mse, running_mae, running_ssim
            )

            training_pbar.set_description(training_info_str)
            training_pbar.close()
            
            validation_pbar = tqdm.tqdm(
                range(len(val_dataloader)), colour="red", desc="Validation Epoch", leave=True
            )
            
            for idx, sample in enumerate(val_dataloader):
                sample = {k: v.to(device) for k, v in sample.items()}
                
                
                with torch.no_grad():

                    # Run the generator forward pass
                    g_loss, mse, mae, ssim = self.g_one_step(sample, "validate")
                    val_g_loss += g_loss.item()
                    val_mse += mse.item()
                    val_mae += mae.item()
                    val_ssim += ssim.item()

                    # Run the discriminator forward pass
                    d_loss = self.d_one_step(n_epoch, idx, sample, "validate")
                    val_d_loss += d_loss.item()


                validation_pbar.update(1)
                
                if self.rank == 0:
                    logger.debug(
                        "val batch idx {:3d}, g_loss:{:7.3f}, d_loss:{:7.3f}".format(
                            idx, g_loss.item(), d_loss.item()
                        )
                    )
            
            val_g_loss /= len(val_dataloader)
            val_d_loss /= len(val_dataloader)
            val_mse /= len(val_dataloader)
            val_mae /= len(val_dataloader)
            val_ssim /= len(val_dataloader)

            val_info_str = "valid_g_loss:{:7.3f}, valid_d_loss:{:7.3f}, valid_mse:{:7.8f}, valid_mae:{:7.4f}, valid_ssim:{:7.8f}".format(
                val_g_loss, val_d_loss, val_mse, val_mae, val_ssim
            )

            validation_pbar.set_description("           " + val_info_str)
            validation_pbar.close()
            
            pbar.update(1)
            if self.rank == 0:
                logger.info(training_info_str + ", " + 
                            val_info_str)
            
            # save the models, overwriting if loss improves
            if (1-val_ssim) < best_loss:
                best_loss = (1-val_ssim)
                torch.save(self.g_net.state_dict(), self.out_dir / "model_gnet_best.pt")
                torch.save(self.d_net.state_dict(), self.out_dir / "model_dnet_best.pt")

            

        return None
