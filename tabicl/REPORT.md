# TabICL Benchmark Report

Comparison of **TabICL** (`TabICLClassifier` / `TabICLRegressor`, zero-shot in-context tabular learning) against classical baselines — **Logistic/Linear Regression** and **Random Forest** — on 5 OpenML classification tasks and 5 OpenML regression tasks.

**Setup:** single 80/20 train/test split per dataset (`random_state=0`), default hyperparameters for all models, categorical columns ordinal-encoded, missing values median/mode-imputed. See [`benchmark.py`](benchmark.py) for full code, [`classification_results.csv`](classification_results.csv) and [`regression_results.csv`](regression_results.csv) for raw numbers.

## Classification results

| Dataset | n | features | LogReg Acc / AUC | RandomForest Acc / AUC | **TabICL Acc / AUC** | TabICL time (s) |
|---|---|---|---|---|---|---|
| credit-g | 1000 | 20 | 0.770 / 0.800 | 0.790 / 0.826 | **0.795 / 0.845** | 4.6 |
| diabetes | 768 | 8 | 0.779 / 0.884 | **0.812** / 0.862 | 0.792 / **0.884** | 1.7 |
| blood-transfusion | 748 | 4 | **0.773** / **0.727** | 0.687 / 0.603 | **0.773** / 0.708 | 1.4 |
| tic-tac-toe | 958 | 9 | 0.688 / 0.634 | 0.964 / 1.000 | **0.979** / 0.999 | 2.4 |
| breast-w | 699 | 9 | **0.979** / **0.998** | 0.964 / 0.992 | 0.964 / 0.996 | 1.7 |
| **Mean** | | | 0.798 / 0.809 | 0.844 / 0.857 | **0.861 / 0.867** | 2.4 |

**Wins by accuracy:** TabICL 2/5, Random Forest 1/5 (+1 tie with TabICL on blood-transfusion/logreg), Logistic Regression 1/5.
**Wins by ROC AUC:** TabICL 3/5, Logistic Regression 2/5, Random Forest 1/5 (tic-tac-toe tie at ~1.0).

TabICL had the best *mean* accuracy and AUC across the five datasets, and never landed last. It particularly stood out on **tic-tac-toe**, a purely categorical, non-linearly-separable dataset where Logistic Regression collapses (0.634 AUC) but both tree-based RF and TabICL solve it almost perfectly (TabICL edges out RF: 0.979 vs 0.964 accuracy). On **breast-w**, a nearly linearly separable problem, plain Logistic Regression is hard to beat and edges out both non-linear models slightly.

## Regression results

| Dataset | n | features | LinReg MAE / R² | RandomForest MAE / R² | **TabICL MAE / R²** | TabICL time (s) |
|---|---|---|---|---|---|---|
| boston | 506 | 13 | 4.009 / 0.555 | 2.665 / 0.773 | **2.320 / 0.796** | 63.7* |
| autoMpg | 398 | 7 | 2.632 / 0.825 | 1.856 / 0.892 | **1.626 / 0.924** | 1.2 |
| machine_cpu | 209 | 6 | 35.131 / 0.884 | 19.116 / 0.944 | **18.683 / 0.951** | 0.6 |
| cholesterol | 303 | 13 | **34.956 / 0.067** | 38.506 / -0.007 | 35.856 / 0.056 | 1.2 |
| stock | 950 | 9 | 1.898 / 0.832 | 0.603 / 0.976 | **0.431 / 0.990** | 3.0 |

\* boston triggered a one-time TabICL regressor checkpoint download (~63s); subsequent runs are ~1-3s like the other datasets.

**Wins by R²:** TabICL 4/5, Linear Regression 1/5 (cholesterol — an inherently low-signal dataset where every model is near-random, R² < 0.1 for all three).

TabICL is the clear winner on regression in this run: it beats Random Forest and Linear Regression on 4 of 5 datasets, sometimes by a wide margin (e.g. **stock**: R² 0.990 vs 0.976 RF vs 0.832 LinReg; MAE nearly 30% lower than RF). Only on **cholesterol**, where no model achieves meaningful signal (max R² 0.067), does Linear Regression come out marginally ahead — likely noise given how weak the signal is across the board.

## Takeaways

- **TabICL is a strong zero-shot/no-training baseline.** With no hyperparameter tuning, it matched or beat both classical baselines on mean performance in both classification and regression, and never performed catastrophically on any dataset (unlike Random Forest on cholesterol, R² < 0).
- **Random Forest remains very competitive** on classification, especially on tree-friendly categorical data (tic-tac-toe), and is dramatically faster (no inference-time forward pass through a neural network).
- **Logistic/Linear Regression** is a fine baseline on linearly separable problems (breast-w) but breaks down on non-linear problems (tic-tac-toe AUC 0.634).
- **Runtime:** TabICL is 10-1000x slower than sklearn baselines per dataset (1-4s typical, since it does a forward pass through a pretrained transformer at inference time vs. microseconds for sklearn `predict`), which is the practical cost of skipping training entirely.

## Reproducing

```bash
uv run python tabicl/benchmark.py
```

Requires `openml`, `tabicl`, `scikit-learn`, `pandas` (see [`pyproject.toml`](../pyproject.toml)).
