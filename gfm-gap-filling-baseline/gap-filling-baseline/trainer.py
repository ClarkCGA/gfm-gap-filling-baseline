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

# ToDo
# Implement and test different learning rates for generator and discriminator and
# do multiple discriminator steps per generator step. See [1] and [2]
#
# [1] Heusel et. al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", 2018
# [2] Zhang et. al., "Self-Attention Generative Adversarial Networks", 2019


class Trainer:
    def __init__(
        self, g_net, d_net, visualization, n_bands, time_steps, generator_lr = 0.00001, discriminator_lr = 0.00004, alpha=4, local_rank=0, out_dir=None  
    ):
        
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

        self.g_optim = torch.optim.Adam(
            self.g_net.parameters(), lr=self.generator_lr, betas=(0, 0.9)
        )
        self.d_optim = torch.optim.Adam(
            self.d_net.parameters(), lr=self.discriminator_lr, betas=(0, 0.9)
        )
        
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.mean_squared_error = MeanSquaredError().to(self.device)
        self.mean_abs_error = MeanAbsoluteError().to(self.device)
        self.structural_similarity = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        
        self.g_loss = loss.HingeGenerator()
        self.d_loss = loss.HingeDiscriminator()

    def visualize_tcc(self, n_epoch, idx, sample, dest_fake, disc_real, disc_fake, cloudmask):

        cloudmask_img = [tensor[0:1,6:7,:,:].clone().detach() for tensor in cloudmask]
        cloudmask_rescale = [torch.nn.functional.interpolate(tensor, size=(224,224), mode='nearest')[0,:,:,:] for tensor in cloudmask_img]
        cloudmask_pad = [torch.nn.functional.pad(tensor, (2,2,2,2), value=1) for tensor in cloudmask_rescale]
        cloudmask_pad = [tensor.expand(3,228,228) for tensor in cloudmask_pad]

        red_tensor = torch.zeros_like(cloudmask_pad[0])
        red_tensor[0] = 1
        
        disc_real_img = [tensor[0:1,:,:,:].clone().detach() for tensor in disc_real]
        disc_real_rescale = [torch.nn.functional.interpolate(tensor, size=(224,224), mode='nearest')[0,:,:,:] for tensor in disc_real_img]
        disc_real_pad = [torch.nn.functional.pad(tensor, (2,2,2,2), value=0) for tensor in disc_real_rescale]
        disc_real_pad = [tensor.expand(3,228,228) for tensor in disc_real_pad]
        disc_real_pad = [tensor1 * tensor2 for tensor1, tensor2 in zip(disc_real_pad, cloudmask_pad)]
        disc_real_pad = [
            torch.where(cloud_tensor == 0, red_tensor, disc_tensor) for disc_tensor, cloud_tensor in zip(disc_real_pad, cloudmask_pad)
        ]

        disc_fake_img = [tensor[0:1,:,:,:].clone().detach() for tensor in disc_fake]
        disc_fake_rescale = [torch.nn.functional.interpolate(tensor, size=(224,224), mode='nearest')[0,:,:,:] for tensor in disc_fake_img]
        disc_fake_pad = [torch.nn.functional.pad(tensor, (2,2,2,2), value=0) for tensor in disc_fake_rescale]
        disc_fake_pad = [tensor.expand(3,228,228) for tensor in disc_fake_pad]
        disc_fake_pad = [tensor1 * tensor2 for tensor1, tensor2 in zip(disc_fake_pad, cloudmask_pad)]
        disc_fake_pad = [
            torch.where(cloud_tensor == 0, red_tensor, disc_tensor) for disc_tensor, cloud_tensor in zip(disc_fake_pad, cloudmask_pad)
        ]

        mean = torch.tensor([495.7316,  814.1386,  924.5740])[:, None, None].to(red_tensor.device).flip(0) # R, G, B
        std = torch.tensor([286.9569, 359.3304, 576.3471])[:, None, None].to(red_tensor.device).flip(0) # R, G, B
        
        masked = []
        generated = []
        unmasked = []
        for t in range(1, self.time_steps+1):
            masked_img = sample["masked"][0,(t-1)*self.n_bands:(t-1)*self.n_bands+3,:,:].clone().flip(0) * 3
            cloud = sample["cloud"][0,(t-1)*self.n_bands:(t-1)*self.n_bands+3,:,:].clone()
            cloud_masked = torch.where(cloud == 1, cloud, masked_img)
            cloud_masked = torch.nn.functional.pad(cloud_masked, (2,2,2,2), value=0)
            masked.append(cloud_masked)
        for t in range(1, self.time_steps+1):
            gen_img = dest_fake[0,(t-1)*self.n_bands:(t-1)*self.n_bands+3,:,:].clone().flip(0) * 3
            masked_img = sample["masked"][0,(t-1)*self.n_bands:(t-1)*self.n_bands+3,:,:].clone().flip(0) * 3
            cloud = sample["cloud"][0,(t-1)*self.n_bands:(t-1)*self.n_bands+3,:,:].clone()
            gen_cloud_masked = gen_img * cloud
            gen_reconstruction = gen_cloud_masked + masked_img
            gen_result = torch.nn.functional.pad(gen_reconstruction, (2,2,2,2), value=0)
            generated.append(gen_result)
        for t in range(1, self.time_steps+1):
            unmasked_img = sample["unmasked"][0,(t-1)*self.n_bands:(t-1)*self.n_bands+3,:,:].clone().flip(0) * 3
            masked_img = sample["masked"][0,(t-1)*self.n_bands:(t-1)*self.n_bands+3,:,:].clone().flip(0) * 3
            unmasked_img += masked_img
            unmasked_img = torch.nn.functional.pad(unmasked_img, (2,2,2,2), value=0)
            unmasked.append(unmasked_img)
        generated[0]=disc_fake_pad[0]
        generated[2]=disc_fake_pad[1]
        unmasked[0]=disc_real_pad[0]
        unmasked[2]=disc_real_pad[1]
        masked = torch.cat(masked, dim=1)
        generated = torch.cat(generated, dim=1)
        unmasked = torch.cat(unmasked, dim=1)
        torchvision.utils.save_image(
            torch.cat([masked]+[generated]+[unmasked], dim=2), self.out_dir/ "epoch{:04}_idx{:04}_gen.jpg".format(n_epoch, idx),
        )

    def g_one_step(self, sample, split):
        self.g_optim.zero_grad()

        g_input = sample["masked"] # Generator input is the masked ground truth

        dest_fake = self.g_net(g_input)
        gen_unmasked = dest_fake * sample["cloud"]
        gen_reconstruction = gen_unmasked + sample["masked"]
        d_output_fake = self.d_net(g_input, gen_reconstruction).final
        cloud_mean = torch.mean(sample["cloud"][:,6:12,:,:])
        
        # Create a list of downscaled cloud masks to weight discriminator outputs
        cloudmask = [torch.nn.functional.max_pool2d(sample["cloud"], 2 ** (3 + scale)) for scale in range(len(d_output_fake))]

        loss_val = sum(self.g_loss(*output) for output in zip(d_output_fake, cloudmask))
        
        # In this implementation of MSE loss, the mean squared error is normalized by the number of masked values we are generating.
        # This ensures that the model is not rewarded for non-generated pixel values.
        mse_normalized = torch.nn.functional.mse_loss(gen_unmasked, sample["unmasked"]) 
        mse_normalized /= torch.mean(sample["cloud"])
        
        mse_score = self.mean_squared_error(gen_unmasked[:,6:12,:,:], sample["unmasked"][:,6:12,:,:])
        mse_score /= cloud_mean

        mae_score = self.mean_abs_error(gen_unmasked[:,6:12,:,:], sample["unmasked"][:,6:12,:,:])
        mae_score /= cloud_mean
        
        ssim = self.structural_similarity(gen_unmasked[:,6:12,:,:], sample["unmasked"][:,6:12,:,:])

        loss_val += self.alpha * mse_normalized
        
        if split == "train":
            loss_val.backward()
            self.g_optim.step()

        return loss_val, mse_score, mae_score, ssim

    def d_one_step(self, n_epoch, idx, sample, split):
        self.d_optim.zero_grad()

        g_input = sample["masked"]


        dest_fake = self.g_net(g_input).detach()
        gen_unmasked = dest_fake * sample["cloud"]
        gen_reconstruction = gen_unmasked + sample["masked"]

        ground_truth = sample["masked"] + sample["unmasked"]

        disc_real = self.d_net(g_input, ground_truth).final
        disc_fake = self.d_net(g_input, gen_reconstruction).final
        cloudmask = [torch.nn.functional.max_pool2d(sample["cloud"], 2 ** (3 + scale)) for scale in range(len(disc_real))]
        loss_val = sum(
            self.d_loss(*disc_out) for disc_out in zip(disc_real, disc_fake, cloudmask)
        )
        
        if idx % 5 == 0 and split == "validate" and self.visualization == "image":
            self.visualize_tcc(n_epoch, idx, sample, dest_fake, disc_real, disc_fake, cloudmask)

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
                sample = {k: v.to(device) for k, v in sample.items()}
                g_loss, mse, mae, ssim = self.g_one_step(sample, "train")
                running_g_loss += g_loss.item()
                running_mse += mse.item()
                running_mae += mae.item()
                running_ssim += ssim.item()

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
            
            if (1-val_ssim) < best_loss:
                best_loss = (1-val_ssim)
                torch.save(self.g_net.state_dict(), self.out_dir / "model_gnet_best.pt")
                torch.save(self.d_net.state_dict(), self.out_dir / "model_dnet_best.pt")

            

        return None
