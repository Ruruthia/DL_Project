"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train, evaluate


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train,
            inputs=["train_data_loader", "val_data_loader", "params:model_type", "params:config",
                    "params:num_epochs", "params:gpus", "params:project"],
            outputs="model_path",
            name="train_model"
        ),
        node(
            func=evaluate,
            inputs=["test_data_loader", "model_path", "params:model_type", "params:gpus", "params:project",
                    "params:output_path"],
            outputs=None,
            name="evaluate_model"
        ),

    ])
