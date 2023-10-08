# gfm-gap-filling-baseline
Baseline model for gap filling as part of the GFM downstream task evaluations

## Directory Structure

Directories should be structured as follows:

```
cgan/
├── gfm-gap-filling-baseline/
│ ├── gap-filling-baseline/
│ │ ├── datasets/
│ │ ├── models/
│ │ ├── options/
│ ├── Dockerfile
│ ├── .git
│ ├── README.md
├── data/
│ ├── results
│ ├── training_data
```

## Running the Docker Image

Navigate to CGAN and run 
```
docker run --gpus all -it -v $PWD:/workspace/ -p 8888:8888 ganfill
```
This will start a Jupyer Lab session which can be accessed via a web browser.

## Running the Fine-Tuning Script

In order to run the fine-tuning script, run train.py on the command line as a python module. For example:

```
python -m train.py --epochs 200 --batch_size 16 --model_cap 64 --dataset gapfill --mask_position 2 --alpha 5 --discriminator_lr 0.0001 --generator_lr 0.0005 --cloud_range 0.01 1.0 --training_length 1600 --local_rank 0
```

**--epochs** is the number of epochs the model will train for.

**--batch_size** can be modified according to memory constraints.

**--model_cap** defines the depth of the generative model in terms of the initial number of feature classes for the first U-Net layer. This can be modified according to memory constraints and complexity of the model.

**--dataset** directs the train.py script to the desired dataset configuration python script in the dataset folder.

**--mask_position** is a list of integers, with each input integer defining a position where a cloud scene should be used to mask the multi-temporal input image. For a three-scene image, an input of 2 would denote the middle time scene. 

**--alpha** defines the relative weight given to mean squared error and hinge loss in updating the generator - the formula is as follows: loss = hinge + alpha * mse

**--discriminator_lr** is a float defining the learning rate of the discriminator.

**--generator_lr** is a float defining the learning rate of the generator.

**--cloud_range** is the lower and upper limits of the ratio of clouds for masks that will be input randomly during training. During validation, the same set of cloud masks are used regardless of inputs for testing consistency across experiments.

**--local_rank** determines which GPU the module will run on. This allows for parallel experiments on machines with multiple GPUs available.

**--training_len** defines the number of time series image chips the model will train on. These will be randomly subsampled from the training set.

## Generating Graphs of Training Performance

Use create_graphs.ipynb to create graphs of model performance during fine-tuning. Replace the variable `job_id` with the experiment whose performance you want to visualize, e.g. `subset_6231_2023-08-20-17:01:03_uneven_bs16`

## Generating Example Images and Per-Image Statistics

These can be run for any weights checkpoint. 

visualize.py is run similarly to train.py. The script will access a checkpoint and save images to a new images directory in the same directory as the checkpoint. For example:

```
python -m visualize --model_cap 64 --batch size --dataset gapfill --dataroot /workspace/data/gapfill6band --mask_position 2 --cloud_range 0.01 1.0 --local_rank 0 --checkpoint_dir subset_6231_2023-08-20-17:01:03_uneven_bs16
```

To create .csv files containing per-image and per-band statistics for the entire validation dataset, run as follows:

```
python -m create_stats --model_cap 64 --batch_size 16 --dataset gapfill --dataroot /workspace/data/gapfill6band --mask_position 2 --cloud_range 0.01 1.0 --local_rank 0 --checkpoint_dir subset_6231_2023-08-20-17:01:03_uneven_bs16
```

## Generating Visualizations of Per-Image Statistics

Use per_image_graphs.ipynb to create visualizations of the distributions and correlations of per-image performance metrics. Replace the variable `job_id` with the experiment whose performance you want to visualize, e.g. `subset_6231_2023-08-20-17:01:03_uneven_bs16`

## Generating Visualizations of Band Correlations

Use band_correlations.ipynb to create visualizations of band correlations for the low-coverage example image and the first 200 images of testing. Replace the variable `job_id` with the experiment whose performance you want to visualize, e.g. `subset_6231_2023-08-20-17:01:03_uneven_bs16`

