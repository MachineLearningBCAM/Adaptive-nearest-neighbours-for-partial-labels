# Data

This folder contains the datasets used in the experiments of our paper.
The actual dataset files are not included in the repository due to their size.
Instructions for obtaining each dataset are provided below.

## Vision Datasets (CIFAR-10, MNIST, Fashion-MNIST)

Pre-extracted GoogLeNet features for the three vision benchmarks are available 
in the [Releases page](../../releases) of this repository. Download the following 
files and place them in this folder:

- `cifar_raw_googlenet.mat` — CIFAR-10 dataset with pre-extracted GoogLeNet features
- `mnist_raw_googlenet.mat` — MNIST dataset with pre-extracted GoogLeNet features  
- `fashion_raw_googlenet.mat` — Fashion-MNIST dataset with pre-extracted GoogLeNet features

Each `.mat` file contains the raw GoogLeNet feature vectors extracted from a 
pre-trained GoogLeNet network, along with the corresponding labels. These files 
are ready to use directly with the provided preprocessing pipeline 
(`nn_preprocess_vision.m`).

## Real-World Partial Label Datasets (MIRFlickr, MSRCv2)

These datasets are hosted by the PALM lab at Southeast University and must be 
downloaded directly from their repository:

**Download from:** https://palm.seu.edu.cn/zhangml/Resources.htm

Download the following files and place them in this folder:

- `mirflickr.mat` — MIRFlickr web image classification dataset
- `MSRCv2.mat` — MSRCv2 object recognition dataset

These datasets provide pre-extracted descriptors and are used with the 
real-world preprocessing pipeline (`nn_preprocess_real_partial_datasets.m`).

## Folder Structure After Setup

After downloading all files, this folder should look like:
```
data/
├── README.md
├── cifar_raw_googlenet.mat
├── fashion_raw_googlenet.mat
├── mnist_raw_googlenet.mat
├── mirflickr.mat
└── MSRCv2.mat
```
