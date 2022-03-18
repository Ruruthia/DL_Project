"""
This is a boilerplate pipeline 'data_split'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split,
            inputs=["params:source", "params:destination", "params:test_ratio", "params:val_ratio", "params:seed"],
            outputs=None,
        )
    ])
