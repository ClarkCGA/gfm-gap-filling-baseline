# gfm-gap-filling-baseline
Baseline model for gap filling as part of the GFM downstream task evaluations

5/25
Currently, the build is configured to synthesize rgb satellite imagery using DEM and segmentation maps of the Democratic Republic of the Congo. 

The command line input for running train.py with the drc data is as follows: python -m train.py --epochs 1 --batch_size 1 --model_cap 64 --lbda 5.0 --num_workers 1 --dataset drc --dataroot ./data --input dem seg --output rgb

Currently, the code is set to crop inputs to 32x32 with batch sizes of 1, but still runs out of CUDA memory at the generator loss step. This needs troubleshooting. For reference, this code runs with 192x192 images on the same computer in an Anaconda environment in Windows. This may have to do with driver issues rather than the code itself.

Make sure you have the data folder in the same host directory as the dockerfile, requirements, and gap-filling-baseline folder. The data folder will be hosted on OneDrive at https://clarkuedu-my.sharepoint.com/:f:/g/personal/dgodwin_clarku_edu/EiHFb9ipP6lKlzl_5uxgEVIBtN0Rv4pPMbhWycTh4WBaFQ?e=YLgvV0

