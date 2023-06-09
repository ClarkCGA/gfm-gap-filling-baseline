# gfm-gap-filling-baseline
Baseline model for gap filling as part of the GFM downstream task evaluations

6/9
The build is configured to fill cloud gaps. It also produces visualizations of generator output during the training process through a command line argument. Next steps are building code to evaluate performance, which can be used to tune the model.

Run in docker using: '''docker run -v "$PWD/data:/workspace/gap-filling-baseline/data" --rm -it --runtime=nvidia --gpus all cgan'''

Then, run: '''python -m train.py --epochs 50 --batch_size 16 --model_cap 16 --dataset gapfill --dataroot ./data/gapfill --mask_position 2 --visualization image'''

The easiest way to modulate the CUDA memory demands of the code is to modify batch size and model capacity.

Make sure that you have the gapfill data folder in its own sub folder in the data directory - e.g. gfm-gap-filling-baseline/data contains gfm-gap-filling-baseline/data/gapfill and gfm-gap-filling-baseline/data/results. The data folder will be hosted on OneDrive at https://clarkuedu-my.sharepoint.com/:f:/g/personal/dgodwin_clarku_edu/EiHFb9ipP6lKlzl_5uxgEVIBtN0Rv4pPMbhWycTh4WBaFQ?e=YaQaO5

