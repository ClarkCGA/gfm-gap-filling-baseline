# gfm-gap-filling-baseline
Baseline model for gap filling as part of the GFM downstream task evaluations

Currently, the build is configured to synthesize rgb satellite imagery using DEM and segmentation maps of the Democratic Republic of the Congo.
The command line input for running train.py with the drc data is as follows:
python -m train.py --epochs 200 --batch_size 1 --model_cap 64 --lbda 5.0 --num_workers 1 --dataset drc --dataroot ./data --input dem seg --output rgb
