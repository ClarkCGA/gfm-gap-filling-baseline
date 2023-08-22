import logging

import torch
import torchvision
import tqdm
import numpy as np
import loss
from PIL import Image


logger = logging.getLogger(__name__)

class Visualizer:
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

    def visualize_tcc(self, idx, sample, dest_fake, disc_real, disc_fake, cloudmask):

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
            torch.cat([masked]+[generated]+[unmasked], dim=2), self.out_dir / "images" / "idx{:04}_gen.jpg".format(idx),
        )

    def d_one_step(self, idx, sample):

        g_input = sample["masked"]
        
        dest_fake = self.g_net(g_input).detach()
        gen_unmasked = dest_fake * sample["cloud"]
        gen_reconstruction = gen_unmasked + sample["masked"]

        ground_truth = sample["masked"] + sample["unmasked"]

        disc_real = self.d_net(g_input, ground_truth).final
        disc_fake = self.d_net(g_input, gen_reconstruction).final
        cloudmask = [torch.nn.functional.max_pool2d(sample["cloud"], 2 ** (3 + scale)) for scale in range(len(disc_real))]

        self.visualize_tcc(idx, sample, dest_fake, disc_real, disc_fake, cloudmask)

    def visualize(self, val_dataloader):
        device = torch.device(f"cuda:{self.local_rank}")
        validation_pbar = tqdm.tqdm(
            range(len(val_dataloader)), colour="red", desc="Visualizations", leave=True
        )
        
        for idx, sample in enumerate(val_dataloader):
            if idx % 5 == 0:
                sample = {k: v.to(device) for k, v in sample.items()}
            
                with torch.no_grad():
                # Run the discriminator forward pass
                    self.d_one_step(idx, sample)
            validation_pbar.update(1)
                    
        return None
