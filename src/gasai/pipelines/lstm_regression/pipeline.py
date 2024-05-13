"""
This is a boilerplate pipeline 'lstm_regression'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import sequence_generate, split_data, scale_sequences, convert_to_tensors, create_dataloaders, train_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=sequence_generate,
                inputs=["s3_conc_aligned_df","params:lstm_model_options"],
                outputs=["lstm_sequences_np","lstm_targets_np"],
                name="lstm_sequences_generate",
            ),
            node(
                func=split_data,
                inputs=["lstm_sequences_np","lstm_targets_np"],
                outputs=["lstm_sequences_train","lstm_sequences_test", "lstm_targets_train", "lstm_targets_test"],
                name="lstm_sequences_split",
            ),
            node(
                func=scale_sequences,
                inputs=["lstm_sequences_train","lstm_sequences_test"],
                outputs=["lstm_sequences_train_scaled","lstm_sequences_test_scaled"],
                name="lstm_sequence_scale",
            ),
            node(
                func=convert_to_tensors,
                inputs=["lstm_sequences_train_scaled","lstm_sequences_test_scaled", "lstm_targets_train", "lstm_targets_test"],
                outputs=["lstm_train_sequences_tensor","lstm_test_sequences_tensor","lstm_train_targets_tensor","lstm_test_targets_tensor"],
                name="lstm_sequence_tensors",
            ),
            node(
                func=create_dataloaders,
                inputs=["lstm_train_sequences_tensor","lstm_train_targets_tensor","lstm_test_sequences_tensor","lstm_test_targets_tensor", "params:lstm_model_options"],
                outputs=["lstm_train_loader","lstm_test_loader"],
                name="lstm_data_loaders",
            ),
            node(
                func=train_model,
                inputs=["lstm_train_loader", "lstm_test_loader", "params:lstm_model_options"],
                outputs=["lstm_model"],
                name="lstm_train",
            ),
        ]
    )
