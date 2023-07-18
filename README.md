# gfm-gap-filling-baseline
Baseline model for gap filling as part of the GFM downstream task evaluations

6/15
The build is configured to fill cloud gaps. It also produces visualizations of generator output during the training process through a command line argument. It can then be used to evaluate the performance of the model and visualize results on a validation dataset.

Run in docker using: ```docker run -v "$PWD:/workspace/" --rm -it --runtime=nvidia --gpus all cgan```

Then, run: ```python -m train.py --epochs 1 --batch_size 16 --model_cap 32 --dataset gapfill --dataroot /workspace/data/gapfill6band --mask_position 2 --visualization image --alpha 0.2```

To test the model, use ```python test.py data/results/{NAME OF RESULTS FILE GENERATED IN TRAINING}/model_gnet.pt```
This will generate visualizations of generator output for the validation dataset as well as the normalized mean squared error of that output as compared to the ground truth, normalized to the number of masked pixels.

The easiest way to modulate the CUDA memory demands of the code is to modify batch size and model capacity.

Make sure that you have the gapfill data folder in its own sub folder in the data directory - e.g. gfm-gap-filling-baseline/data contains gfm-gap-filling-baseline/data/gapfill and gfm-gap-filling-baseline/data/results. The data folder will be hosted on OneDrive at https://clarkuedu-my.sharepoint.com/:f:/g/personal/dgodwin_clarku_edu/EiHFb9ipP6lKlzl_5uxgEVIBtN0Rv4pPMbhWycTh4WBaFQ?e=YaQaO5

