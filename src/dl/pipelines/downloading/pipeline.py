"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""
from kedro.pipeline import Pipeline, pipeline, node

from .nodes import download_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=download_data,
            inputs=None,
            outputs="Sharks_dataset",
            name="download_data_node"
        ),
        node(
            func=split,
            inputs="Sharks",
            outputs="Sharks_dataset",
            name="download_data_node"
        )
    ])
