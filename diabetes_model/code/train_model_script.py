import argparse
import os
from argparse import Namespace

import mlflow
import pandas as pd
import yaml

from .train_model import CrossValidationData, TrainTestData, train_model


def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-source", type=str)
    parser.add_argument("--num-hyperopt-evals", type=int)
    parser.add_argument("--num-hyperopt-trials-to-log", type=int)

    args = parser.parse_args()

    return args


def load_training_data(data_source) -> TrainTestData:
    X_train_file = os.path.join(data_source, "training", "X_train.pkl")
    X_train = pd.read_pickle(X_train_file)

    y_train_file = os.path.join(data_source, "training", "y_train.pkl")
    y_train = pd.read_pickle(y_train_file)

    X_train_upsample_file = os.path.join(
        data_source, "training", "X_train_upsample.pkl"
    )
    X_train_upsample = pd.read_pickle(X_train_upsample_file)

    y_train_upsample_file = os.path.join(
        data_source, "training", "y_train_upsample.pkl"
    )
    y_train_upsample = pd.read_pickle(y_train_upsample_file)

    X_test_file = os.path.join(data_source, "training", "X_test.pkl")
    X_test = pd.read_pickle(X_test_file)

    y_test_file = os.path.join(data_source, "training", "y_test.pkl")
    y_test = pd.read_pickle(y_test_file)

    train_test_data = TrainTestData(
        X_train, y_train, X_train_upsample, y_train_upsample, X_test, y_test
    )
    return train_test_data


def load_cv_data(data_source, num_cv: int = 5) -> CrossValidationData:
    X_train_cv_list: list[pd.DataFrame] = []
    X_train_cv_upsample_list: list[pd.DataFrame] = []
    y_train_cv_list: list[pd.Series] = []
    y_train_cv_upsample_list: list[pd.Series] = []
    X_val_cv_list: list[pd.DataFrame] = []
    y_val_cv_list: list[pd.DataFrame] = []
    for i in range(num_cv):
        # X train cross validation data
        X_train_cv_file = os.path.join(
            data_source, "training", "cv", f"X_train_{i}.pkl"
        )
        X_train_cv = pd.read_pickle(X_train_cv_file)
        X_train_cv_list.append(X_train_cv)

        # Upsampled X train cross validation data
        X_train_cv_upsample_file = os.path.join(
            data_source, "training", "cv", f"X_train_{i}_upsample.pkl"
        )
        X_train_cv_upsample = pd.read_pickle(X_train_cv_upsample_file)
        X_train_cv_upsample_list.append(X_train_cv_upsample)

        # y train cross validation
        y_train_cv_file = os.path.join(
            data_source, "training", "cv", f"y_train_{i}.pkl"
        )
        y_train_cv = pd.read_pickle(y_train_cv_file)
        y_train_cv_list.append(y_train_cv)

        # Upsampled y train cross validation
        y_train_cv_upsample_file = os.path.join(
            data_source, "training", "cv", f"y_train_{i}_upsample.pkl"
        )
        y_train_cv_upsample = pd.read_pickle(y_train_cv_upsample_file)
        y_train_cv_upsample_list.append(y_train_cv_upsample)

        # X validation for cross validation
        X_val_cv_file = os.path.join(data_source, "training", "cv", f"X_val_cv{i}.pkl")
        X_val_cv = pd.read_pickle(X_val_cv_file)
        X_val_cv_list.append(X_val_cv)

        # y validation for cross validation
        y_val_cv_file = os.path.join(data_source, "training", "cv", f"y_val_cv{i}.pkl")
        y_val_cv = pd.read_pickle(y_val_cv_file)
        y_val_cv_list.append(y_val_cv)

    cross_validation_data = CrossValidationData(
        X_train_cv_list,
        y_train_cv_list,
        X_train_cv_upsample_list,
        y_train_cv_upsample_list,
        X_val_cv_list,
        y_val_cv_list,
    )
    return cross_validation_data


if __name__ == "__main__":
    args = parse_arguments()

    data_source = args.data_source
    num_hyperopt_evals = args.num_hyperopt_evals
    num_hyperopt_trials_to_log = args.num_hyperopt_trials_to_log

    data_params_yaml_file = os.path.join(data_source, "data_params.yaml")
    with open(data_params_yaml_file, "r") as f:
        data_params_dict = yaml.safe_load(f)
    num_cv = data_params_dict["xgboost_training"]["num_cross_validation_sets"]

    train_test_data = load_training_data(data_source)
    cross_validation_data = load_cv_data(data_source)
    mlflow.set_experiment("diabetes_prediction")

    with mlflow.start_run():
        mlflow.set_tag(key="isParentRun", value=1)
        train_model(
            train_test_data,
            cross_validation_data,
            num_hyperopt_evals,
            num_hyperopt_trials_to_log,
        )
