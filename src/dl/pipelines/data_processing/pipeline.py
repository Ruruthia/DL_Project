"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import process_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=process_data,
            inputs=["params:split_source", "params:batch_size"],
            outputs=["data_loaders"],
            name="process_data_node"
        )
    ])
