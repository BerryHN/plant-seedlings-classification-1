# [Plant Seedlings Classification](https://www.kaggle.com/c/plant-seedlings-classification)

Can you differentiate a weed from a crop seedling?

The ability to do so effectively can mean better crop yields and better stewardship of the environment.

The Aarhus University Signal Processing group, in collaboration with University of Southern Denmark, has recently released a dataset containing images of approximately 960 unique plants belonging to 12 species at several growth stages.

This repository contains scripts to retrain a VGG19 model on the dataset and implementation of customized `f1-score` in `keras`.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

The packages you need to install:

```
numpy
pandas
h5py
keras
tensorflow
Pillow
scikit-learn
```

### Preparing data

First, create a folder to store your dataset.

```bash
mkdir datasets
```

Then, go to [this page](https://www.kaggle.com/c/landmark-recognition-challenge/data) to download the dataset and unzip the zipped files. Store all the datasets inside the `datasets` folder and you will have something like below:

```
../datasets/sample_submission.csv
../datasets/train/
../datasets/test/
```

Lastly, run the script `generate_dataframe.py` to generate `train.csv` that contains three columns, i.e. `file`, `species`, and `species_id`.

```
python generate_dataframe.py
```

### Training

You can immediately start the training using the command below:

```
cd scripts/
python training.py
```

Optional arguments:
```
--output_dir : path to output directory
--dataset : path to the csv file generated using generate_dataframe.py
--n_splits : Number of stratified k fold splits
--batch_size : Size per training batch
--epochs : number of epochs for training
```

The model and the weight trained will be stored as:

```
model: plant_vgg19.h5
weights: plant_vgg19_weights.h5
```

The training and validation scores are stored in `log.json`.

## Authors

* [**Hardian Lawi**](https://github.com/hardianlawi)

## Acknowledgments

I extend my appreciation to the Aarhus University Department of Engineering Signal Processing Group for hosting the original data and Kaggle for hosting the competition.
