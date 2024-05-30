from collections import defaultdict
from dataclasses import dataclass
from functools import partial

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll.base import scope
from xgboost import XGBClassifier

from .metrics import score_classifier
from .visualizations import create_classification_report_visual


@dataclass
class TrainTestData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_train_upsample: pd.DataFrame
    y_train_upsample: pd.Series
    X_test: pd.DataFrame
    y_test: pd.DataFrame


@dataclass
class CrossValidationData:
    X_train_cv_list: list[pd.DataFrame]
    y_train_cv_list: list[pd.DataFrame]
    X_train_upsample_cv_list: list[pd.DataFrame]
    y_train_upsample_cv_list: list[pd.Series]
    X_val_cv_list: list[pd.DataFrame]
    y_val_cv_list: list[pd.DataFrame]


def hyperopt_optimize_function(
    space: dict[str, any],
    X_train_list_cv: list[pd.DataFrame],
    y_train_list_cv: list[pd.Series],
    X_val_list_cv: list[pd.DataFrame],
    y_val_list_cv: list[pd.Series],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, any]:
    val_metrics_cv: dict[str, any] = defaultdict(lambda: [])
    for X_train_cv, y_train_cv, X_val_cv, y_val_cv in zip(
        X_train_list_cv, y_train_list_cv, X_val_list_cv, y_val_list_cv
    ):
        clf = XGBClassifier(
            **space,
            eval_metric="auc",
            early_stopping_rounds=20,
        )
        clf.fit(
            X_train_cv,
            y_train_cv,
            eval_set=[(X_val_cv, y_val_cv)],
            verbose=False,
        )
        beta = 2
        eval_metrics = score_classifier(
            clf=clf,
            X=X_val_cv,
            y=y_val_cv,
            beta=beta,
        )

        for metric_name, metric_val in eval_metrics.items():
            val_metrics_cv[metric_name].append(metric_val)
    clf = XGBClassifier(
        **space,
        early_stopping_rounds=20,
        eval_metric="auc",
    )
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    test_metrics = score_classifier(
        clf,
        X=X_test,
        y=y_test,
        beta=beta,
    )

    avg_val_metrics = {key: np.mean(vals) for key, vals in val_metrics_cv.items()}

    return {
        "loss": -np.mean(avg_val_metrics["f_score"]),
        "status": STATUS_OK,
        "model": clf,
        "params": space,
        "avg eval metrics": avg_val_metrics,
        "test metrics": test_metrics,
    }


def mlflow_log_best_trials(
    trials: Trials,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    num_trials_to_log: int = 5,
):
    """From the hyperopt trials, logs the num_trials_to_log best models, ranked
    according to their losses

    Args:
        trials (Trials): hyperopt trials
        num_trials_to_log (int, optional): number of trials to log. Defaults to 5.
    """
    sorted_trials = sorted(trials, key=lambda x: x["result"]["loss"])
    for i in range(num_trials_to_log):
        with mlflow.start_run(nested=True):
            params = sorted_trials[i]["result"]["params"]
            avg_eval_metrics = sorted_trials[i]["result"]["avg eval metrics"]
            test_metrics = sorted_trials[i]["result"]["test metrics"]
            clf = sorted_trials[i]["result"]["model"]

            for metric_name, metric_val in avg_eval_metrics.items():
                mlflow.log_metric(key=f"{metric_name}_eval", value=metric_val)
            for metric_name, metric_val in test_metrics.items():
                mlflow.log_metric(key=f"{metric_name}_test", value=metric_val)

            for param_name, param_value in params.items():
                mlflow.log_param(key=param_name, value=param_value)
            fig_classification_report = create_classification_report_visual(
                clf, X_test, y_test
            )
            mlflow.log_figure(
                figure=fig_classification_report,
                artifact_file="classification_report.png",
            )
            plt.close("all")
            mlflow.xgboost.log_model(clf, "model")


def train_model(
    train_test_data: TrainTestData,
    cross_validation_data: CrossValidationData,
    num_hyperopt_evals: int,
    num_hyperopt_trials_to_log: int,
):
    space = {
        "eta": hp.uniform("eta", 0, 1.0),
        "max_depth": scope.int(hp.quniform("max_depth", 2, 20, 1)),
        "subsample": hp.uniform("subsample", 0, 1),
        "colsample_bytree": hp.uniform("colsample_bytree", 0, 1),
        "n_estimators": scope.int(hp.quniform("n_estimators", 20, 5000, 25)),
        "gamma": hp.uniform("gamma", 0, 0.3),
        "min_child_weight": hp.uniform("min_child_weight", 0, 1),
        "nthread": 4,
    }

    X_train_upsample_cv_list = cross_validation_data.X_train_upsample_cv_list
    y_train_upsample_cv_list = cross_validation_data.y_train_upsample_cv_list
    X_val_cv_list = cross_validation_data.X_val_cv_list
    y_val_cv_list = cross_validation_data.y_val_cv_list

    X_train_upsample = train_test_data.X_train_upsample
    y_train_upsample = train_test_data.y_train_upsample
    X_test = train_test_data.X_test
    y_test = train_test_data.y_test

    # Find optimal hyperparameters
    func_to_optimize = partial(
        hyperopt_optimize_function,
        X_train_list_cv=X_train_upsample_cv_list,
        y_train_list_cv=y_train_upsample_cv_list,
        X_val_list_cv=X_val_cv_list,
        y_val_list_cv=y_val_cv_list,
        X_train=X_train_upsample,
        y_train=y_train_upsample,
        X_test=X_test,
        y_test=y_test,
    )

    # Find good hyperparameters, log the hyperopt trials
    trials = Trials()
    rstate = np.random.default_rng(42)
    _ = fmin(
        fn=func_to_optimize,
        space=space,
        algo=tpe.suggest,
        max_evals=num_hyperopt_evals,
        trials=trials,
        rstate=rstate,
    )

    mlflow_log_best_trials(
        trials,
        X_test,
        y_test,
        num_hyperopt_trials_to_log,
    )

    return
