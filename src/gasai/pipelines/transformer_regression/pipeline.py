"""
This is a boilerplate pipeline 'transformer_regression'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import prepare_data, create_ResistanceDataset, wrap_loader, train_model, sliding_data


# def create_pipeline(**kwargs) -> Pipeline:
#     return pipeline(
#         [
#             node(
#                 func=prepare_data,
#                 inputs=["s3_conc_aligned_df","params:transformer_model_options"],
#                 outputs=["trans_scaled_train_features", "trans_train_targets", "trans_scaled_val_features", "trans_val_targets"],
#                 name="trans_sequences_generate",
#             ),
#             node(
#                 func=create_ResistanceDataset,
#                 inputs=["trans_scaled_train_features","trans_train_targets","trans_scaled_val_features","trans_val_targets"],
#                 outputs=["trans_train_dataset","trans_val_dataset"],
#                 name="trans_train_dataset_generate",
#             ),
#             node(
#                 func=wrap_loader,
#                 inputs=["trans_train_dataset","trans_val_dataset","params:transformer_model_options"],
#                 outputs=["trans_train_loader","trans_val_loader"],
#                 name="trans_train_val_loader",
#             ),
#             node(
#                 func=train_model,
#                 inputs=["trans_train_loader","trans_val_loader","params:transformer_model_options"],
#                 outputs=["trans_model"],
#                 name="trans_train_model",
#             ),
#         ]
#     )

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=sliding_data,
                inputs=["s3_conc_aligned_df","params:transformer_model_options"],
                outputs=["trans_seq_train_loader", "trans_seq_test_loader"],
                name="trans_sliding_data",
            ),
            node(
                func=train_model,
                inputs=["trans_seq_train_loader","trans_seq_test_loader","params:transformer_model_options"],
                outputs=["trans_model"],
                name="trans_train_model",
            ),
        ]
    )