# Sharks classification (ongoing)

## Overview
(will be updated during the project)

The goal of the project is to deliver a deep learning model classifying an open-source dataset of [Shark species](https://www.kaggle.com/larusso94/shark-species) available on Kaggle.

The project will consist of a training and evaluation scripts wrapped with [Kedro](https://kedro.readthedocs.io/en/stable/index.html) project. We are going to prepare a fully transferable setup so that you can train and run the model either locally or using a cloud provider. We have decided to use Google Cloud Platform, so you will find detailed instructions how to start it off with GCP.

Therefore, we are going to use some state-of-the-art convolutional neural networks adjusted to the needs of the dataset. We are aware that the project is not revolutionary at, but its goal is to learn how to deliver end-to-end ML model rather than make an innovative step in research.

The papers which describe the models that we are going to use are obviously:
- A. Krizhevsky, I. Sutskever, and G. Hinton. [Imagenet classification with deep convolutional neural networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf). In NIPS, 2012.
- K. Simonyan and A. Zisserman. [Very deep convolutional networks for large-scale image recognition](https://arxiv.org/pdf/1409.1556.pdf). In ICLR, 2015.
- K. He, X. Zhang, S. Ren, and J. Sun. [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf). In CVPR, 2016.

## How to install dependencies

Declare any dependencies in `src/requirements.txt` for `pip` installation and `src/environment.yml` for `conda` installation.

To install them, run:

```
kedro install
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to train the model and log the progress

After preparing the data with `kedro run`, use the `notebooks/experiments.ipynb` notebook to train the model, log training progress and save the best version of the model.
Logging is handled by Weights&Biases. To see the results, you need to register [here](https://wandb.ai/).
In the following weeks training is going to be scripted, but for experiments we preferred to use Jupyter notebook.
