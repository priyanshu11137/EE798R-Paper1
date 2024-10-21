# DM-Count: Distribution Matching for Crowd Counting

This repository implements DM-Count, a novel crowd counting method using Optimal Transport (OT) to match predicted and ground-truth density distributions.

## Running the Code

### Preprocess the dataset:
```
python preprocess_dataset.py --dataset <dataset_name> --input-dataset-path <input_dir> --output-dataset-path <output_dir>
```
Supported dataset names: qnrf, nwpu

### Train the model:
```
python train.py --dataset <dataset_name> --data-dir <data_path> --device <gpu_id>
```
Supported dataset names: qnrf, sha, shb, nwpu

### Test the model:
```
python test.py --model-path <model_path> --data-path <data_dir> --dataset <dataset_name>
```
Supported dataset names: qnrf, sha, shb, nwpu

### Demo run on single image:
```
python demo.py
```

## Recommended Execution Environments

- QNRF: Google Colab (upload dataset to Google Drive)
- Shanghai Tech Part A and B: Kaggle (dataset pre-loaded)
- Demo: Kaggle or Google Colab (avoid local execution due to potential computational demands)

Note: Kaggle and Colab links with execution details and results are provided in the accompanying PDF.

## Datasets

- [QNRF Dataset](https://www.crcv.ucf.edu/data/ucf-qnrf/)
- [Shanghai Tech Part A and B](https://www.kaggle.com/tthien/shanghaitech)

## Pre-trained Models

Pre-trained models are available but may not fully reproduce ablation study results due to computational constraints.

## Official Implementation

The original implementation can be found at: https://github.com/cvlab-stonybrook/DM-Count

This repository aims to reproduce results and provide insights into Density Distribution Analysis techniques.
