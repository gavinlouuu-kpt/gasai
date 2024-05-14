"""
This is a boilerplate pipeline 'lstm_regression'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import train_model, sliding_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=sliding_data,
                inputs=["s3_conc_aligned_df","params:lstm_model_options"],
                outputs=["lstm_train_loader", "lstm_test_loader"],
                name="lstm_sliding_data",
            ),
            node(
                func=train_model,
                inputs=["lstm_train_loader","lstm_test_loader","params:lstm_model_options"],
                outputs=["lstm_model"],
                name="lstm_train_model",
            ),
        ]
    )
