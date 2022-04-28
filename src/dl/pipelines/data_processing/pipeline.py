"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split, process_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split,
            inputs=["sharks_raw_dataset", "params:test_ratio", "params:val_ratio", "params:seed"],
            outputs="sharks_data_subsets"
        ),
        node(
            func=process_data,
            inputs=["sharks_data_subsets", "params:batch_size"],
            outputs=["train_data_loader", "test_data_loader", "val_data_loader"],
            name="process_data_node"
        )
    ])
