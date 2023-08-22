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

class Stat_Creator:
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
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.mean_squared_error = MeanSquaredError()
        self.mean_abs_error = MeanAbsoluteError()
        self.structural_similarity = StructuralSimilarityIndexMeasure(data_range=1.0)

    def g_one_step(self, sample):
        g_input = sample["masked"] # Generator input is the masked ground truth

        dest_fake = self.g_net(g_input)
        gen_unmasked = dest_fake * sample["cloud"]
        gen_reconstruction = gen_unmasked + sample["masked"]
        d_output_fake = self.d_net(g_input, gen_reconstruction).final

        data_dicts = []
        
        for b in range(g_input.size(dim=0)):
            cloud_mean = torch.mean(sample["cloud"][b:b+1,6:12,:,:].detach().cpu())

            mse_score = self.mean_squared_error(gen_unmasked[b:b+1,6:12,:,:].detach().cpu(), sample["unmasked"][b:b+1,6:12,:,:].detach().cpu())
            mse_score /= cloud_mean


            mae_score = self.mean_abs_error(gen_unmasked[b:b+1,6:12,:,:].detach().cpu(), sample["unmasked"][b:b+1,6:12,:,:].detach().cpu())
            mae_score /= cloud_mean
        
            ssim = self.structural_similarity(gen_unmasked[b:b+1,6:12,:,:].detach().cpu(), sample["unmasked"][b:b+1,6:12,:,:].detach().cpu())

            per_band_mse_list = []
            per_band_mae_list = []
            per_band_ssim_list = []

            for n in range(6): # For each band of 6, do the following:
                # Get the MSE for only that band, selected with n, from the predicted and input, masked with the cloud mask.
                per_band_mse = self.mean_squared_error(gen_unmasked[b:b+1,6+n:7+n,:,:].detach().cpu(), sample["unmasked"][b:b+1,6+n:7+n,:,:].detach().cpu())
                # Adjust the mse by the proportion of masked pixels.
                per_band_mse /= cloud_mean
                # Append to the list of per band mse for this batch
                per_band_mse_list.append(per_band_mse.item())
                # Get the MAE for only that band, selected with n, from the predicted and input, masked with the cloud mask.
                per_band_mae = self.mean_abs_error(gen_unmasked[b:b+1,6+n:7+n,:,:].detach().cpu(), sample["unmasked"][b:b+1,6+n:7+n,:,:].detach().cpu())
                # Adjust the mae by the proportion of masked pixels.
                per_band_mae /= cloud_mean
                # Append to the list of per band mae for this batch
                per_band_mae_list.append(per_band_mae.item())
                # Get the SSIM for only that band at the middle time step.
                per_band_ssim_score = self.structural_similarity(gen_unmasked[b:b+1,6+n:7+n,:,:].detach().cpu(), sample["unmasked"][b:b+1,6+n:7+n,:,:].detach().cpu())
                # Append to the list of per band SSIM for this batch
                per_band_ssim_list.append(per_band_ssim_score.item())


            # Append a dictionary representing this batch's stats, to be compiled into a dataframe
            data_dict = {'Overall SSIM': ssim.item(), 
                         'Overall MSE': mse_score.item(),
                         'Overall MAE': mae_score.item(),
                         'Mask Ratio': cloud_mean.item(),
                         'B02 MSE': per_band_mse_list[0],
                         'B03 MSE': per_band_mse_list[1],
                         'B04 MSE': per_band_mse_list[2],
                         'B05 MSE': per_band_mse_list[3],
                         'B07 MSE': per_band_mse_list[4],
                         'B08 MSE': per_band_mse_list[5],
                         'B02 MAE': per_band_mae_list[0],
                         'B03 MAE': per_band_mae_list[1],
                         'B04 MAE': per_band_mae_list[2],
                         'B05 MAE': per_band_mae_list[3],
                         'B07 MAE': per_band_mae_list[4],
                         'B08 MAE': per_band_mae_list[5],
                         'B02 SSIM': per_band_ssim_list[0],
                         'B03 SSIM': per_band_ssim_list[1],
                         'B04 SSIM': per_band_ssim_list[2],
                         'B05 SSIM': per_band_ssim_list[3],
                         'B07 SSIM': per_band_ssim_list[4],
                         'B08 SSIM': per_band_ssim_list[5],
                         }

            data_dicts.append(data_dict)
        

        return data_dicts
        
    def stats(self, val_dataloader):
        
        
        validation_pbar = tqdm.tqdm(
            range(len(val_dataloader)), colour="red", desc="Stats", leave=True
        )
        data_list = []
        for idx, sample in enumerate(val_dataloader):
            sample = {k: v.to(self.device) for k, v in sample.items()}

            data_dicts = self.g_one_step(sample)
            data_list.extend(data_dict for data_dict in data_dicts)
            validation_pbar.update(1)
                    
        return data_list
