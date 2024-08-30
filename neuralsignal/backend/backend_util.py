import logging
import mlflow
import os
import pickle
from pathlib import Path
from neuralsignal.core.modules.neuralsignal_config import sdk_config

logging.basicConfig(level=sdk_config.logging_level())


def save_to_mlflow(
        model, mlflow_uri, experiment_name, run_name=None,
        register_model=False
        ):
    logging.info(f"Saving model to MLflow experiment {experiment_name}")
    if run_name is None:
        run_name = model.model_name
    mlflow.set_tracking_uri(mlflow_uri)
    try:
        mlflow.set_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        pass

    mlflow_run = mlflow.start_run(
        run_name=run_name,
        description=model['description'],
    )

    with mlflow_run:

        # Log metrics, params, and tags
        mlflow.log_metrics(model['metrics'])
        mlflow.log_params(model['params'])
        mlflow.set_tags(model['tags'])

        if register_model:
            # Log model
            registered_model_name = f"{experiment_name.split('/')[-1]}"
            mlflow_model_info = mlflow.sklearn.log_model(
                sk_model=model.model,
                artifact_path="S1",
                registered_model_name=registered_model_name,
            )
        else:
            mlflow_model_info = None

        # Log figures
        if "figures" in model.config:
            for fig_name, fig in model.config['figures'].items():
                mlflow.log_figure(fig, f"{fig_name}.png")

        # Log artifacts
        if "artifacts" in model.config:
            for artifact_name, artifact in model.config['artifacts'].items():
                if isinstance(artifact, dict):
                    mlflow.log_dict(artifact, artifact_name + ".json")
                else:
                    mlflow.log_artifact(artifact, artifact_name)

    return mlflow_model_info


def load_s1_model_from_mlflow(model_id):
    raise NotImplementedError


def count_files_in_dir(dir_path: str, extension: str) -> int:
    ct = 0
    for f in os.listdir(dir_path):
        if extension in f:
            ct += 1
    return ct


def save_scan_to_disk(scan, directory_path):
    os.makedirs(directory_path, exist_ok=True)
    _id = str(scan["_id"])
    fp = f"{directory_path}/{_id}.scan"
    with open(fp, 'wb') as handle:
        pickle.dump(scan, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_scan_from_disk(scan_id, directory_path):
    fp = f"{directory_path}/{str(scan_id)}.scan"
    with open(fp, 'rb') as handle:
        retVal = pickle.load(handle)
    return retVal


def reduce_hd_cache(mng, size: int, directory_path):
    curr_size = count_files_in_dir(directory_path, ".scan")
    while curr_size > size:
        df = delete_oldest_file(directory_path)
        _id = Path(df).stem
        mng.scan_hd_cache.remove(_id)
        curr_size = count_files_in_dir(directory_path, ".scan")


def get_list_of_files(
        directory_path, extension="scan", stem_only=True):
    list_of_all_files = os.listdir(directory_path)
    if not stem_only:
        list_of_files = [x for x in list_of_all_files if f".{extension}" in x]
    else:
        list_of_files =\
            [Path(x).stem for x in list_of_all_files if f".{extension}" in x]
    return list_of_files


def delete_oldest_file(directory_path, extension="scan"):
    list_of_all_files = os.listdir(directory_path)
    list_of_files = [x for x in list_of_all_files if f".{extension}" in x]
    full_path = [f"{directory_path}/{x}" for x in list_of_files]
    oldest_file = min(full_path, key=os.path.getctime)
    os.remove(oldest_file)
    return oldest_file
