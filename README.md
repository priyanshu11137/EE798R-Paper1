# DM-Count: Distribution Matching for Crowd Counting

This repository contains the implementation of DM-Count, a novel method for crowd counting that leverages Optimal Transport (OT) to match predicted and ground-truth density distributions.

## Running the Code

### Preprocess the dataset:
```
python preprocess_dataset.py --dataset <dataset name: qnrf or nwpu> --input-dataset-path <original data directory> --output-dataset-path <processed data directory>
```

### Train the Dataset:
```
python train.py --dataset <dataset name: qnrf, sha, shb or nwpu> --data-dir <path to dataset> --device <gpu device id>
```
### Test the Model:
```
python test.py --model-path <path of the model to be evaluated> --data-path <directory for the dataset> --dataset <dataset name: qnrf, sha, shb or nwpu>
```

## Datasets
The method has been tested on the following datasets:
+ QNRF can be downloaded [here](https://www.crcv.ucf.edu/data/ucf-qnrf/)
+ Shanghai Tech Part A and Part B can be downloaded [here](https://www.kaggle.com/tthien/shanghaitech)

## Pre-Trained Models
The method has been tested on the following datasets:
+ QNRF can be downloaded [here](https://www.crcv.ucf.edu/data/ucf-qnrf/)
+ Shanghai Tech Part A and Part B can be downloaded [here](https://www.kaggle.com/tthien/shanghaitech)

## Official Implementation
The official GitHub repository mentioned in the paper is: https://github.com/cvlab-stonybrook/DM-Count

This repository is a rewrite of the project with the aim to reproduce the results and gain a better understanding of Density Distribution Analysis techniques.

