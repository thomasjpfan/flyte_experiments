import mlflow
import json
from pathlib import Path
from functools import partial
import xgboost as xgb
import numpy as np
import optuna
from mlflow.models import infer_signature
from optuna.study.study import ObjectiveFuncType
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from flytekit import task, workflow, ImageSpec, current_context, Deck
from flytekitplugins.deck.renderer import MarkdownRenderer
from flytekit.types.file import FlyteFile
from optuna.integration.mlflow import MLflowCallback


xgboost_spec = ImageSpec(
    name="flyte_playground",
    base_image="ghcr.io/flyteorg/flytekit:py3.11-1.10.1",
    packages=["mlflow", "scikit-learn", "xgboost", "pandas", "optuna"],
    python_version="3.11",
    registry="ghcr.io/thomasjpfan",
    platform="linux/arm64",
)


# TODO: Use dataframe with Structured Dataset
@task(cache=True, cache_version="2", container_image=xgboost_spec)
def get_data() -> (np.ndarray, np.ndarray):
    return load_breast_cancer(return_X_y=True)


def _objective(
    trial: ObjectiveFuncType,
    X_train: np.ndarray,
    X_valid: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    param = {
        "verbosity": 0,
        "objective": "binary:logitraw",
        "tree_method": "exact",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }
    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        )

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical(
            "sample_type", ["uniform", "weighted"]
        )
        param["normalize_type"] = trial.suggest_categorical(
            "normalize_type", ["tree", "forest"]
        )
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    bst = xgb.train(param, dtrain)
    preds = bst.predict(dvalid)
    pred_labels = (preds > 0).astype(np.int32)
    accuracy = accuracy_score(y_valid, pred_labels)
    return accuracy


@task(container_image=xgboost_spec, enable_deck=True)
def search_for_best_params(
    X: np.ndarray,
    y: np.ndarray,
    mlflow_tracking_uri: str,
    experiment_name: str,
    n_trials: int,
) -> (FlyteFile, float, str):
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    objective = partial(
        _objective, X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid
    )

    context = current_context()
    execution_id = context.execution_id

    workflow_url = (
        "http://localhost:30080/console"
        f"/projects/{execution_id.project}/"
        f"domains/{execution_id.domain}/"
        f"executions/{execution_id.name}"
    )

    description = f"[Flyte Workflow Link]({workflow_url})"

    mlflow.set_tracking_uri(uri=mlflow_tracking_uri)
    experiment = mlflow.set_experiment(experiment_name=experiment_name)

    with mlflow.start_run(
        run_name=str(execution_id.name),
        description=description,
        tags={"workflow_url": workflow_url},
    ) as run:
        mlflc = MLflowCallback(
            tracking_uri=mlflow_tracking_uri,
            metric_name="accuracy",
            create_experiment=False,
            mlflow_kwargs={"nested": True},
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=600, callbacks=[mlflc])

    experiment_url = f"{mlflow_tracking_uri}/#/experiments/{experiment.experiment_id}"
    Deck("MLFlow", MarkdownRenderer().to_html(f"## [MLFlow url]({experiment_url})"))

    best_trial = study.best_trial
    best_params = best_trial.params

    context = current_context()
    json_path = Path(context.working_directory) / "best_params.json"

    with json_path.open("w") as f:
        json.dump(best_params, f)

    return json_path, best_trial.value, run.info.run_id


@task(container_image=xgboost_spec, enable_deck=True)
def train_full_model(
    mlflow_tracking_uri: str,
    experiment_name: str,
    run_id: str,
    best_params_path: FlyteFile,
    best_metric: float,
    X: np.ndarray,
    y: np.ndarray,
) -> (FlyteFile, str):
    with open(best_params_path, "r") as f:
        params = json.load(f)

    dtrain = xgb.DMatrix(X, label=y)
    bst = xgb.train(params, dtrain)

    signature = infer_signature(X, y)

    context = current_context()
    execution_id = context.execution_id
    workflow_url = (
        "http://localhost:30080/console"
        f"/projects/{execution_id.project}/"
        f"domains/{execution_id.domain}/"
        f"executions/{execution_id.name}"
    )

    description = f"[Flyte Workflow Link]({workflow_url})"

    mlflow.set_tracking_uri(uri=mlflow_tracking_uri)
    experiment = mlflow.set_experiment(experiment_name=experiment_name)

    with mlflow.start_run(
        run_id=run_id,
        description=description,
        tags={"workflow_url": workflow_url},
    ) as run:
        mlflow.xgboost.log_model(bst, "model", signature=signature)
        mlflow.log_metric("accuracy", best_metric)
        mlflow.log_params(params)

    context = current_context()
    model_path = Path(context.working_directory) / "best_model.json"
    bst.save_model(model_path)

    mlflow_url = (
        f"{mlflow_tracking_uri}/#/experiments/"
        f"{experiment.experiment_id}/runs/{run.info.run_id}"
    )

    Deck("MLFlow", MarkdownRenderer().to_html(f"## [MLFlow url]({mlflow_url})"))

    return (model_path, mlflow_url)


@workflow
def run_workflow(mlflow_tracking_uri: str, experiment_name: str, n_trials: int):
    X, y = get_data()
    best_params, best_metric, run_id = search_for_best_params(
        X=X,
        y=y,
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        n_trials=n_trials,
    )
    train_full_model(
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        run_id=run_id,
        best_params_path=best_params,
        best_metric=best_metric,
        X=X,
        y=y,
    )
