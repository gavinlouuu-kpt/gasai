"""
This is a boilerplate pipeline 'lstm_regression'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import sequence_generate, split_data, scale_sequences, convert_to_tensors, create_dataloaders, setup_training

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=sequence_generate,
                inputs=["s3_conc_aligned_df","params:lstm_model_options"],
                outputs=["sequences_np","targets_np"],
                name="sequences_generate",
            ),
            node(
                func=split_data,
                inputs=["sequences_np","targets_np"],
                outputs=["sequences_train","sequences_test", "targets_train", "targets_test"],
                name="sequences_split",
            ),
            node(
                func=scale_sequences,
                inputs=["sequences_train","sequences_test"],
                outputs=["sequences_train_scaled","sequences_test_scaled"],
                name="sequence_scale",
            ),
            node(
                func=convert_to_tensors,
                inputs=["sequences_train_scaled","sequences_test_scaled", "targets_train", "targets_test"],
                outputs=["train_sequences_tensor","test_sequences_tensor","train_targets_tensor","test_targets_tensor"],
                name="sequence_tensors",
            ),
            node(
                func=create_dataloaders,
                inputs=["train_sequences_tensor","train_targets_tensor","test_sequences_tensor","test_targets_tensor", "params:lstm_model_options"],
                outputs=["train_sequences_tensor","test_sequences_tensor","train_targets_tensor","test_targets_tensor"],
                name="sequence_tensors",
            ),
        ]
    )
