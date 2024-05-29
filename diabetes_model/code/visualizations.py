import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier


def create_confusion_matrix_visual(
    clf: XGBClassifier, X_test: pd.DataFrame, y_test: pd.DataFrame
) -> Figure:
    return
    # y_pred = clf.
