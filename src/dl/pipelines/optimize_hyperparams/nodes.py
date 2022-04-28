"""
This is a boilerplate pipeline 'optimize_hyperparams'
generated using Kedro 0.17.7
"""
import json

import pytorch_lightning as pl
from dl.models.resnet import ResNet18Lit
from dl.models.vanilla_cnn import CNNLit
from pytorch_lightning.loggers import WandbLogger
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
import ray


def optimize_hyperparams(train_data_loader, val_data_loader, model_type, hyperparams_config, num_samples,
                         num_epochs, cpus, gpus, project, hyperparams_path):
    ray.init()

    trainable = tune.with_parameters(
        train_model,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        model_type=model_type,
        gpus=gpus,
        num_epochs=num_epochs,
        project=project
    )
    config = {
        "lr": tune.uniform(hyperparams_config["min_lr"], hyperparams_config["max_lr"]),
        "momentum": tune.uniform(hyperparams_config["min_momentum"], hyperparams_config["max_momentum"]),
        "gamma": tune.uniform(hyperparams_config["min_gamma"], hyperparams_config["max_gamma"]),
        "step_size": tune.choice(hyperparams_config["step_size"]),
    }

    analysis = tune.run(
        trainable,
        resources_per_trial={
            # "cpu": cpus,
            "gpu": gpus
        },
        local_dir='/home/marys/Documents/Sharks_classification/logs/',
        config=config,
        num_samples=num_samples,
        scheduler=ASHAScheduler(metric="acc", mode="max"),
        name="tune_cnn")

    with open(hyperparams_path, 'w') as f:
        json.dump(analysis.get_best_config("acc", "max"), f)

    ray.shutdown()


def train_model(config, train_data_loader, val_data_loader, model_type, gpus, num_epochs, project):
    if model_type == "resnet":
        model = ResNet18Lit(config)
    elif model_type == "cnn":
        model = CNNLit(config)
    else:
        raise NotImplementedError("Possible model types are resnet or cnn!")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_accuracy',
        dirpath='/home/marys/Documents/Sharks_classification/data/06_models/',
        filename=f'{model_type}' + '-{epoch:02d}-{val_accuracy:.2f}',
        save_top_k=1,
        mode='max')

    metrics = {"loss": "val_loss", "acc": "val_accuracy"}
    tune_callback = TuneReportCallback(metrics, on="validation_end")

    trainer = pl.Trainer(WandbLogger(save_dir="/home/marys/Documents/Sharks_classification/logs/", project=project),
                         gpus=gpus, max_epochs=num_epochs, callbacks=[checkpoint_callback, tune_callback])

    trainer.fit(model, train_data_loader, val_data_loader)
