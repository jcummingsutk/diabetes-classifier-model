import argparse
from argparse import Namespace


def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-folds", type=int)
    parser.add_argument("--train-data")
    parser.add_argument("--test-data")
    parser.add_argument("--target-column")
    parser.add_argument("--num-hyperopt-evals", type=int)
    parser.add_argument("--num-hyperopt-trials-to-log", type=int)

    args = parser.parse_args()

    return args
