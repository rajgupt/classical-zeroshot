"""
Benchmark TabICL against classical baselines (Logistic/Random Forest) on
5 OpenML classification tasks and 5 OpenML regression tasks.

Usage:
    python tabicl/benchmark.py
"""
import time
import warnings

import numpy as np
import openml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from tabicl import TabICLClassifier, TabICLRegressor

warnings.filterwarnings("ignore")

RANDOM_STATE = 0
TEST_SIZE = 0.2

CLASSIFICATION_DATASETS = [
    "credit-g",
    "diabetes",
    "blood-transfusion-service-center",
    "tic-tac-toe",
    "breast-w",
]

REGRESSION_DATASETS = [
    "boston",
    "autoMpg",
    "machine_cpu",
    "cholesterol",
    "stock",
]


def load_openml_dataset(name: str):
    """Fetch an OpenML dataset by name and return X (DataFrame), y (Series)."""
    dataset = openml.datasets.get_dataset(
        name,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=False,
    )
    X, y, categorical_mask, _ = dataset.get_data(target=dataset.default_target_attribute)
    return X, y, categorical_mask


def preprocess(X: pd.DataFrame, categorical_mask):
    """Encode categorical columns as ordinal ints and impute missing values.

    TabICL and sklearn's linear/tree models all expect numeric arrays, so
    categorical columns are ordinal-encoded (fine for tree/ICL models; a
    reasonable, uniform choice across all baselines in this benchmark).
    """
    X = X.copy()
    cat_cols = [col for col, is_cat in zip(X.columns, categorical_mask) if is_cat]
    num_cols = [col for col in X.columns if col not in cat_cols]

    if cat_cols:
        X[cat_cols] = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        ).fit_transform(X[cat_cols].astype(str))

    if num_cols:
        X[num_cols] = SimpleImputer(strategy="median").fit_transform(X[num_cols])
    if cat_cols:
        X[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X[cat_cols])

    return X.to_numpy(dtype=float)


def run_classification_task(name: str) -> dict:
    print(f"\n[classification] {name}")
    X, y, categorical_mask = load_openml_dataset(name)
    X = preprocess(X, categorical_mask)
    y = OrdinalEncoder().fit_transform(np.asarray(y).reshape(-1, 1)).ravel().astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    results = {"dataset": name, "n_samples": X.shape[0], "n_features": X.shape[1]}
    is_binary = len(np.unique(y)) == 2

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        ),
        "random_forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "tabicl": TabICLClassifier(random_state=RANDOM_STATE),
    }

    for model_name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        elapsed = time.time() - start

        acc = accuracy_score(y_test, y_pred)
        if is_binary:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        else:
            y_proba = model.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr")

        results[f"{model_name}_accuracy"] = acc
        results[f"{model_name}_roc_auc"] = auc
        results[f"{model_name}_time_s"] = elapsed
        print(f"  {model_name:22s} acc={acc:.3f} auc={auc:.3f} time={elapsed:.1f}s")

    return results


def run_regression_task(name: str) -> dict:
    print(f"\n[regression] {name}")
    X, y, categorical_mask = load_openml_dataset(name)
    X = preprocess(X, categorical_mask)
    y = np.asarray(y, dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    results = {"dataset": name, "n_samples": X.shape[0], "n_features": X.shape[1]}

    models = {
        "linear_regression": make_pipeline(StandardScaler(), LinearRegression()),
        "random_forest": RandomForestRegressor(random_state=RANDOM_STATE),
        "tabicl": TabICLRegressor(random_state=RANDOM_STATE),
    }

    for model_name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        elapsed = time.time() - start

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[f"{model_name}_mae"] = mae
        results[f"{model_name}_r2"] = r2
        results[f"{model_name}_time_s"] = elapsed
        print(f"  {model_name:22s} mae={mae:.3f} r2={r2:.3f} time={elapsed:.1f}s")

    return results


def main():
    clf_results = [run_classification_task(name) for name in CLASSIFICATION_DATASETS]
    reg_results = [run_regression_task(name) for name in REGRESSION_DATASETS]

    clf_df = pd.DataFrame(clf_results)
    reg_df = pd.DataFrame(reg_results)

    clf_df.to_csv("tabicl/classification_results.csv", index=False)
    reg_df.to_csv("tabicl/regression_results.csv", index=False)

    print("\n=== Classification results ===")
    print(clf_df.to_string(index=False))
    print("\n=== Regression results ===")
    print(reg_df.to_string(index=False))


if __name__ == "__main__":
    main()
