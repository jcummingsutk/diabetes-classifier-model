import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import classification_report
from xgboost import XGBClassifier


def create_classification_report_visual(
    clf: XGBClassifier, X_test: pd.DataFrame, y_test: pd.DataFrame
) -> Figure:
    y_true = y_test
    y_pred = clf.predict(X_test)
    report = classification_report(
        y_true,
        y_pred,
        target_names=[
            "No Diabetes",
            "Diabetes",
        ],
        output_dict=True,
    )
    report = {key: val for key, val in report.items() if key != "accuracy"}
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
    fig.tight_layout()
    sns.heatmap(
        pd.DataFrame(
            {key: report[key] for key in ["No Diabetes", "Diabetes", "weighted avg"]}
        ).T,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1.0,
        ax=ax,
    )
    return fig
