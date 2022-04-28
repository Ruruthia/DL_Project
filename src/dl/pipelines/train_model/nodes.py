"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.17.7
"""
from pathlib import Path

import numpy as np
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import confusion_matrix

from dl.models.resnet import ResNet18Lit
from dl.models.vanilla_cnn import CNNLit

import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
import pandas as pd
import seaborn as sns


def train(train_data_loader, val_data_loader, model_type, config, num_epochs, gpus, project):
    if model_type == "resnet":
        model = ResNet18Lit(config)
    elif model_type == "cnn":
        model = CNNLit(config)
    else:
        raise NotImplementedError("Possible model types are resnet or cnn!")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_accuracy',
        dirpath='data/06_models/',
        filename=f'{model_type}' + '-{epoch:02d}-{val_accuracy:.2f}',
        save_top_k=1,
        mode='max')

    trainer = pl.Trainer(logger=WandbLogger(save_dir=f"logs/", project=project),
                         gpus=gpus, max_epochs=num_epochs, callbacks=[checkpoint_callback])

    trainer.fit(model, train_data_loader, val_data_loader)

    return checkpoint_callback.best_model_path


def evaluate(test_data_loader, model_path, model_type, gpus, project, output_path):

    model = create_model(model_type, config)

    trainer = pl.Trainer(gpus=gpus, logger=WandbLogger(save_dir=f"logs/", project=project,
                                                       name=f'{Path(model_path).stem}_evaluation'))

    trainer.test(model, test_data_loader)

    class_names = test_data_loader.dataset.dataset.classes
    draw_confusion_matrix(model, test_data_loader, class_names,
                          f'{output_path}/{Path(model_path).stem}_confusion_matrix.png')
    draw_incorrectly_classified_examples(model, test_data_loader, class_names,
                                         f'{output_path}/{Path(model_path).stem}_misclassified.png')


def draw_confusion_matrix(model, dataloader, class_names, path):
    labels = []
    predicted = []
    for x, y in dataloader:
        labels.extend(y.data.detach().numpy())
        predicted.extend(list(torch.max(model(x), 1)[1].numpy()))
    cf = confusion_matrix(labels, predicted)
    df_cm = pd.DataFrame(cf, index=class_names,
                         columns=class_names)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True)
    plt.savefig(path)


def draw_incorrectly_classified_examples(model, dataloader, class_names, path):
    imgs = []
    labels = []
    predicted = []
    for x, y in dataloader:
        _, preds = torch.max(model(x), 1)
        incorrect_preds = preds != y.data
        labels.extend(y[incorrect_preds].data.detach().numpy())
        predicted.extend(list((preds[incorrect_preds]).numpy()))
        imgs.extend(x[incorrect_preds])

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    for i in range(3):
        for j in range(3):
            axes[i][j].imshow(np.clip(imgs[3 * i + j].permute(1, 2, 0).numpy(), 0.0, 1.0))
            axes[i][j].set_xlabel(
                f"Correct: {class_names[labels[3 * i + j]]}, predicted: {class_names[predicted[3 * i + j]]}")
    plt.show()
    plt.savefig(path)
