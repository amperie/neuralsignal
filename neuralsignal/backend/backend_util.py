import logging
import mlflow
import os
from neuralsignal.core.modules.neuralsignal_config import sdk_config

logging.basicConfig(level=sdk_config.logging_level())


def save_to_mlflow(model, mlflow_uri, experiment_name):
    logging.info(f"Saving model to MLflow experiment {experiment_name}")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    mlflow_run = mlflow.start_run(
        run_name=model.model_name,
        description=model['description'],
    )

    with mlflow_run:

        # Log metrics, params, and tags
        mlflow.log_metrics(model['metrics'])
        mlflow.log_params(model['params'])
        mlflow.set_tags(model['tags'])

        # Log model
        registered_model_name = f"{experiment_name}"
        mlflow_model_info = mlflow.sklearn.log_model(
            sk_model=model.model,
            artifact_path="S1",
            registered_model_name=registered_model_name,
        )

        # Log figures
        for fig_name, fig in model['figures'].items():
            mlflow.log_figure(fig, f"{fig_name}.png")

    return mlflow_model_info


def count_files_in_dir(dir_path: str, extension: str) -> int:
    ct = 0
    for f in os.listdir():
        if extension in f:
            ct += 1
    return ct
