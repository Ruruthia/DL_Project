"""
This is a boilerplate pipeline 'optimize_hyperparams'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import optimize_hyperparams


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=optimize_hyperparams,
            inputs=["train_data_loader", "val_data_loader", "params:hp_model_type", "params:hp_config",
                    "params:hp_num_samples", "params:hp_num_epochs", "params:hp_cpus", "params:hp_gpus",
                    "params:hp_project", "params:hp_output_path"],
            outputs=None,
            name="optimize_hyperparams"
        ),

    ])
