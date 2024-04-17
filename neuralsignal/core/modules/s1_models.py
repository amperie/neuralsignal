import mlflow


def load_model_from_mlflow(mlflow_uri: str, model_path: str):
    mlflow.set_tracking_uri(mlflow_uri)
    model = mlflow.sklearn.load_model(model_path)
    return model
