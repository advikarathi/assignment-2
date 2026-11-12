"""Discovery: compare a low-code workflow (PyCaret) with a manual one (scikit-learn).

Business problem: a Portuguese bank runs phone campaigns selling term deposits.
Predict, before calling, whether a client will subscribe (y = yes/no) so agents
can prioritise the clients most likely to say yes.

Run with:  uv run discovery.py
"""

import shutil
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no GUI windows; everything is saved to disk
import matplotlib.pyplot as plt
import pandas as pd

IMAGES = Path("images")
IMAGES.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
df = pd.read_csv("data/bank.csv")
# 'duration' (length of the last call) is only known after the call ends, and
# it nearly determines the outcome, so it would leak the answer. Drop it.
df = df.drop(columns=["duration"])
print(f"Data: {df.shape[0]} rows, {df.shape[1] - 1} features")
print(df["y"].value_counts(normalize=True).round(3).to_dict())

TARGET = "y"
CATEGORICAL = ["job", "marital", "education", "default", "housing",
               "loan", "contact", "month", "poutcome"]
NUMERIC = ["age", "balance", "day", "campaign", "pdays", "previous"]


def save_table_png(table: pd.DataFrame, path: Path, title: str) -> None:
    """Render a DataFrame as an image (the PyCaret comparison table)."""
    fig, ax = plt.subplots(figsize=(14, 0.3 * len(table) + 0.6))
    ax.axis("off")
    ax.set_title(title, fontsize=12, loc="left")
    widths = [0.24] + [0.76 / (table.shape[1] - 1)] * (table.shape[1] - 1)
    tbl = ax.table(cellText=table.values, colLabels=table.columns,
                   colWidths=widths, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. PyCaret workflow
# ---------------------------------------------------------------------------
from pycaret.classification import compare_models, plot_model, pull, setup

t0 = time.perf_counter()

setup(data=df, target=TARGET, session_id=42, verbose=False)
top3 = compare_models(n_select=3)
comparison = pull()
best = top3[0]

plot_model(best, plot="confusion_matrix", save=True)

pycaret_seconds = time.perf_counter() - t0

print("\nPyCaret comparison table:")
print(comparison.to_string())
comparison.to_csv(IMAGES / "pycaret_comparison.csv")
save_table_png(comparison.reset_index(drop=True).round(4), IMAGES / "pycaret_comparison.png",
               "PyCaret compare_models() - 10-fold CV on the training set")
shutil.move("Confusion Matrix.png", IMAGES / "pycaret_confusion_matrix.png")
print(f"\nTop 3 models: {[type(m).__name__ for m in top3]}")
print(f"PyCaret section took {pycaret_seconds:.1f} s")

# ---------------------------------------------------------------------------
# 2. scikit-learn workflow: rebuild PyCaret's best model (RidgeClassifier) by hand
# ---------------------------------------------------------------------------
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

t0 = time.perf_counter()

X = df[NUMERIC + CATEGORICAL]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocess = ColumnTransformer([
    ("num", numeric_pipe, NUMERIC),
    ("cat", categorical_pipe, CATEGORICAL),
])

model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", RidgeClassifier(random_state=42)),
])
model.fit(X_train, y_train)

print("\nscikit-learn classification report (held-out 30% test set):")
print(classification_report(y_test, model.predict(X_test), digits=4))

joblib.dump(model, "model.joblib")
sklearn_seconds = time.perf_counter() - t0
print(f"scikit-learn section took {sklearn_seconds:.1f} s; saved model.joblib")
