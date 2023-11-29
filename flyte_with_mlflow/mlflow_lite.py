from flytekit import task, workflow, current_context
import mlflow
from mlflow.models import infer_signature

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

from flytekit import ImageSpec

mlflow_spec = ImageSpec(
    name="flyte_playground",
    base_image="ghcr.io/flyteorg/flytekit:py3.11-1.10.1",
    packages=["mlflow", "scikit-learn"],
    python_version="3.11",
    registry="ghcr.io/thomasjpfan",
)


@task(container_image=mlflow_spec, cache=True, cache_version="1")
def load_data() -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


@task(container_image=mlflow_spec, cache=True, cache_version="1")
def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "multi_class": "auto",
        "random_state": 8888,
    }

    # Train the model
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    return lr


@task(container_image=mlflow_spec)
def evaluate_model(
    model: LogisticRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mlflow_tracking_uri: str,
) -> str:
    context = current_context()
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.set_tracking_uri(uri=mlflow_tracking_uri)
    experiment = mlflow.set_experiment("MLflow Scikit-learn")

    execution_id = context.execution_id

    workflow_url = (
        "http://localhost:30080/console"
        f"/projects/{execution_id.project}/"
        f"domains/{execution_id.domain}/"
        f"executions/{execution_id.name}"
    )

    params = model.get_params()

    artifact_link = f"[Flyte Workflow Link]({workflow_url})"

    # Start an MLflow run
    with mlflow.start_run(
        run_name=str(execution_id.name),
        description=artifact_link,
    ) as run:
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.set_tag("Training Info", "Basic LR model for iris data")
        signature = infer_signature(X_test, y_pred)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="iris_model",
            signature=signature,
            input_example=X_test,
            registered_model_name="tracking-quickstart",
        )

    return (
        f"{mlflow_tracking_uri}/#/experiments/"
        f"{experiment.experiment_id}/runs/{run.info.run_id}"
    )


@workflow
def mlflow_workflow(mlflow_tracking_uri: str) -> str:
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train=X_train, y_train=y_train)
    return evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )
