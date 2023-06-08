# gfm-gap-filling-baseline
Baseline model for gap filling as part of the GFM downstream task evaluations

5/25
Currently, the build is configured to synthesize rgb satellite imagery using DEM and segmentation maps of the Democratic Republic of the Congo. 

Run using:
docker run -v "$PWD/data:/workspace/gap-filling-baseline/data" --rm -it --runtime=nvidia --gpus all cgan

The command line input for running train.py with the drc data is as follows: 
python -m train.py --epochs 1 --batch_size 16 --model_cap 16 --num_workers 1 --dataset drc --dataroot ./data --input dem seg --output rgb

The easiest way to modulate the CUDA memory demands of the code is to modify batch size and model capacity.

Make sure you have the data folder in the same host directory as the dockerfile, requirements, and gap-filling-baseline folder. The data folder will be hosted on OneDrive at https://clarkuedu-my.sharepoint.com/:f:/g/personal/dgodwin_clarku_edu/EiHFb9ipP6lKlzl_5uxgEVIBtN0Rv4pPMbhWycTh4WBaFQ?e=YaQaO5

