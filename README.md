# Sharks classification

## Overview

The goal of the project is to deliver a deep learning model classifying an open-source dataset of [Shark species](https://www.kaggle.com/larusso94/shark-species) available on Kaggle.

The project will consist of a training and evaluation scripts wrapped with [Kedro](https://kedro.readthedocs.io/en/stable/index.html) project. 

Therefore, we are going to use some state-of-the-art convolutional neural networks adjusted to the needs of the dataset. We are aware that the project is not revolutionary at, but its goal is to learn how to deliver end-to-end ML model rather than make an innovative step in research.

The papers which describe the models that we are going to use are obviously:
- A. Krizhevsky, I. Sutskever, and G. Hinton. [Imagenet classification with deep convolutional neural networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf). In NIPS, 2012.
- K. Simonyan and A. Zisserman. [Very deep convolutional networks for large-scale image recognition](https://arxiv.org/pdf/1409.1556.pdf). In ICLR, 2015.
- K. He, X. Zhang, S. Ren, and J. Sun. [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf). In CVPR, 2016.

## How to install dependencies

Due to complicated dependencies, we are using conda environment for this project. 
To prepare it, run:

```
conda env create -f environment.yml
```

## How to run your Kedro pipeline

There are 3 pipelines: 
- `data_processing` for preparing the data
- `train_model` for training and evaluating model with given hyperparameters
- `optimize_hyperparams` for hyperparams optimization using Raytune

To run any of them, run:

```
kedro run --pipeline pipeline_name
```

`data_processing` pipeline will be automatically started before running `train_model` or `optimize_hyperparams`
Results of training (in both `train_model` and `optimize_hyperparams`) will get logged to Weights & Biases.

## Results of experiments

Results of our experiments can be found [here](https://github.com/Ruruthia/DL_Project/blob/123b8b8892a9fc1386de1904f70ad5222f67c2ac/docs/Summary.md).
