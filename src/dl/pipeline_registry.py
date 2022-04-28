"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline
from .pipelines import data_split, data_processing, train_model, optimize_hyperparams



def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    data_processing_pipeline = data_processing.create_pipeline()
    model_train_pipeline = train_model.create_pipeline()
    optimize_hyperparams_pipeline = optimize_hyperparams.create_pipeline()

    return {
        "__default__": pipeline([data_split_pipeline, data_processing_pipeline, model_train_pipeline]),
        "data_split": data_split_pipeline,
        "data_processing": pipeline([data_split_pipeline, data_processing_pipeline]),
        "train_model": pipeline([data_split_pipeline, data_processing_pipeline, model_train_pipeline]),
        "optimize_hyperparams": pipeline([data_split_pipeline, data_processing_pipeline, optimize_hyperparams_pipeline]),
    }
