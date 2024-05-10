"""
This is a boilerplate pipeline 'cnn_regression'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import prepare_data, create_ResistanceDataset, wrap_loader, train_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_data,
                inputs=["s3_conc_aligned_df","params:cnn_model_options"],
                outputs=["scaled_train_features", "train_targets", "scaled_val_features", "val_targets"],
                name="sequences_generate",
            ),
            node(
                func=create_ResistanceDataset,
                inputs=["scaled_train_features","train_targets","scaled_val_features","val_targets"],
                outputs=["train_dataset","val_dataset"],
                name="train_dataset_generate",
            ),
            node(
                func=wrap_loader,
                inputs=["train_dataset","val_dataset","params:cnn_model_options"],
                outputs=["train_loader","val_loader"],
                name="train_val_loader",
            ),
            node(
                func=train_model,
                inputs=["train_loader","val_loader","params:cnn_model_options"],
                outputs=["cnn_model"],
                name="train_model",
            ),
        ]
    )
