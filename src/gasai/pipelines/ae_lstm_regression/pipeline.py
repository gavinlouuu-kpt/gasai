"""
This is a boilerplate pipeline 'ae_lstm_regression'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import ae_init_train, lstm_init_train


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=ae_init_train,
                inputs=["s3_conc_aligned_df","params:ae_lstm_model_options"],
                outputs=["autoencoder", 
                         "ae_seq_train",
                         "ae_seq_test",
                         "ae_targets_train", 
                         "ae_targets_test"],
                name="ae_init_train",
            ),
            node(
                func=lstm_init_train,
                inputs=["autoencoder", 
                        "ae_seq_train",
                        "ae_seq_test",
                        "ae_targets_train", 
                        "ae_targets_test",
                        "params:ae_lstm_model_options"],
                outputs=["ae_lstm_model"],
                name="ae_lstm_train_model",
            ),
        ]
    )
