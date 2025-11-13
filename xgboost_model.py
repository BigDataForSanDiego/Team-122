import argparse
import os
from pathlib import Path
from typing import List

MPL_DIR = Path.cwd() / ".matplotlib_cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
MPL_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

REQUIRED_COLUMNS = [
    "year_month",
    "homeless_count",
    "average_temp",
    "ZORI",
    "unemployment_rate",
    "evictions",
    "covid",
    "precipitation",
]

MAX_LAG_MONTHS = 6
LAGGED_HOMELESS_COLS = [f"homeless_count_lag{lag}" for lag in range(1, MAX_LAG_MONTHS + 1)]

BASE_FEATURE_COLUMNS = [
    "year_month_ordinal",
    "average_temp",
    "ZORI",
    "unemployment_rate",
    "evictions",
    "covid",
    "precipitation",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + LAGGED_HOMELESS_COLS

DEFAULT_TRAIN_RATIO = 0.8


def parse_float_list(values: List[str]) -> List[float]:
    return [float(value) for value in values]


def parse_int_list(values: List[str]) -> List[int]:
    return [int(value) for value in values]


def prepare_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing_cols = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_cols:
        raise ValueError(f"Input data missing required columns: {', '.join(missing_cols)}")

    df["year_month"] = pd.to_datetime(df["year_month"])
    df = df.sort_values("year_month").drop_duplicates(subset="year_month").reset_index(drop=True)

    numeric_cols = [col for col in REQUIRED_COLUMNS if col != "year_month"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both").ffill().bfill()

    if len(df) < MAX_LAG_MONTHS + 1:
        raise ValueError(
            f"Need at least {MAX_LAG_MONTHS + 1} months of data after cleaning, but received {len(df)} rows."
        )

    df["year_month_ordinal"] = df["year_month"].map(lambda dt: dt.toordinal())
    for lag in range(1, MAX_LAG_MONTHS + 1):
        df[f"homeless_count_lag{lag}"] = df["homeless_count"].shift(lag)

    df["homeless_count_delta"] = df["homeless_count"] - df["homeless_count_lag1"]

    df = df.dropna(subset=LAGGED_HOMELESS_COLS + ["homeless_count_delta"]).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/test XGBoost regressor on homelessness data.")
    parser.add_argument("--input", default="cut_off_end.csv", help="Input CSV path (default: cut_off_end.csv)")
    parser.add_argument(
        "--output",
        default="xgb_predictions_covid.csv",
        help="Where to save XGBoost predictions (default: xgb_predictions_covid.csv)",
    )
    parser.add_argument(
        "--actual-vs-pred-plot",
        default="xgb_actual_vs_pred_covid.png",
        help="Path to save actual vs predicted plot with train/test panels",
    )
    parser.add_argument(
        "--feature-importance-plot",
        default="xgb_feature_importance_covid.png",
        help="Path to save feature importance plot",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Override number of initial months to use for training (default: derived from train-ratio)",
    )
    parser.add_argument(
        "--train-start",
        type=str,
        default=None,
        help="Optional YYYY-MM start month to include (rows before this date are dropped)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help=f"Proportion of data to allocate to training when train-size is unspecified (default: {DEFAULT_TRAIN_RATIO:.2f})",
    )
    parser.add_argument(
        "--target-mode",
        choices=["level", "delta"],
        default="delta",
        help="Predict raw homeless counts ('level') or month-over-month change ('delta'). Default: delta.",
    )
    parser.add_argument(
        "--n-estimators",
        nargs="+",
        default=["300", "600"],
        help="Candidate n_estimators for XGBoost (default: 300 600)",
    )
    parser.add_argument(
        "--max-depth",
        nargs="+",
        default=["3", "4"],
        help="Candidate max_depth values (default: 3 4)",
    )
    parser.add_argument(
        "--learning-rate",
        nargs="+",
        default=["0.03", "0.05"],
        help="Candidate learning rates (default: 0.03 0.05)",
    )
    parser.add_argument(
        "--subsample",
        nargs="+",
        default=["0.8"],
        help="Candidate subsample ratios (default: 0.8)",
    )
    parser.add_argument(
        "--colsample",
        nargs="+",
        default=["0.8"],
        help="Candidate colsample_bytree ratios (default: 0.8)",
    )
    parser.add_argument(
        "--min-child-weight",
        nargs="+",
        default=["1", "5"],
        help="Candidate min_child_weight values (default: 1 5)",
    )
    parser.add_argument(
        "--gamma",
        nargs="+",
        default=["0", "0.5"],
        help="Candidate gamma (minimum loss reduction) values (default: 0 0.5)",
    )
    parser.add_argument(
        "--reg-lambda",
        nargs="+",
        default=["1", "5"],
        help="Candidate L2 regularization values (default: 1 5)",
    )
    parser.add_argument(
        "--reg-alpha",
        nargs="+",
        default=["0", "0.5"],
        help="Candidate L1 regularization values (default: 0 0.5)",
    )
    parser.add_argument(
        "--random-state",
        nargs="+",
        default=["42"],
        help="Candidate random states for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    df = prepare_dataframe(Path(args.input))

    if args.train_start:
        train_start_ts = pd.to_datetime(args.train_start)
        df = df[df["year_month"] >= train_start_ts].reset_index(drop=True)
        if len(df) < MAX_LAG_MONTHS + 1:
            raise ValueError("Filtering by train-start left too few rows for training.")

    if args.train_size is not None:
        if args.train_size <= 0 or args.train_size >= len(df):
            raise ValueError("train-size must be positive and leave at least one month for testing.")
        train_size = args.train_size
    else:
        inferred_size = int(len(df) * args.train_ratio)
        train_size = min(max(inferred_size, 1), len(df) - 1)
        if train_size <= 0 or train_size >= len(df):
            raise ValueError("Unable to derive a valid train/test split from train-ratio.")

    feature_cols = FEATURE_COLUMNS
    X = df[feature_cols]
    target_column = "homeless_count_delta" if args.target_mode == "delta" else "homeless_count"
    y = df[target_column]
    dates = df["year_month"]
    y_counts = df["homeless_count"]

    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    y_train_counts, y_test_counts = y_counts.iloc[:train_size], y_counts.iloc[train_size:]
    dates_train, dates_test = dates.iloc[:train_size], dates.iloc[train_size:]

    param_grid = {
        "n_estimators": parse_int_list(args.n_estimators),
        "max_depth": parse_int_list(args.max_depth),
        "learning_rate": parse_float_list(args.learning_rate),
        "subsample": parse_float_list(args.subsample),
        "colsample_bytree": parse_float_list(args.colsample),
        "min_child_weight": parse_float_list(args.min_child_weight),
        "gamma": parse_float_list(args.gamma),
        "reg_lambda": parse_float_list(args.reg_lambda),
        "reg_alpha": parse_float_list(args.reg_alpha),
        "random_state": parse_int_list(args.random_state),
    }

    model = XGBRegressor(
        objective="reg:squarederror",
        booster="gbtree",
        n_jobs=1,
        tree_method="hist",
        eval_metric="rmse",
    )
    n_splits = min(5, max(2, train_size - 1))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    grid = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    best_model: XGBRegressor = grid.best_estimator_

    y_train_pred_raw = best_model.predict(X_train)
    if args.target_mode == "delta":
        y_train_pred = X_train["homeless_count_lag1"].to_numpy() + y_train_pred_raw
    else:
        y_train_pred = y_train_pred_raw
    train_rmse = mean_squared_error(y_train_counts, y_train_pred) ** 0.5
    train_r2 = r2_score(y_train_counts, y_train_pred)

    if len(X_test) > 0:
        y_test_pred_raw = best_model.predict(X_test)
        if args.target_mode == "delta":
            y_test_pred = X_test["homeless_count_lag1"].to_numpy() + y_test_pred_raw
        else:
            y_test_pred = y_test_pred_raw
        test_rmse = mean_squared_error(y_test_counts, y_test_pred) ** 0.5
        test_r2 = r2_score(y_test_counts, y_test_pred)
    else:
        y_test_pred = pd.Series(dtype=float)
        test_rmse = float("nan")
        test_r2 = float("nan")

    train_df = pd.DataFrame(
        {
            "year_month": dates_train.dt.strftime("%Y-%m-%d"),
            "split": "train",
            "y_true": y_train_counts.values,
            "y_pred": y_train_pred,
        }
    )
    test_df = pd.DataFrame(
        {
            "year_month": dates_test.dt.strftime("%Y-%m-%d"),
            "split": "test",
            "y_true": y_test_counts.values,
            "y_pred": y_test_pred,
        }
    )
    predictions_df = pd.concat([train_df, test_df], ignore_index=True)
    predictions_df.to_csv(args.output, index=False)

    print("Best Params:", grid.best_params_)
    print(f"Train RMSE: {train_rmse:.3f} | R2: {train_r2:.3f}")
    if len(X_test) > 0:
        print(f" Test RMSE: {test_rmse:.3f} | R2: {test_r2:.3f}")
    print(f"Saved predictions to {args.output}")

    num_panels = 2 if len(X_test) > 0 else 1
    fig, axes = plt.subplots(num_panels, 1, figsize=(11, 8), sharex=True)
    if num_panels == 1:
        axes = [axes]

    axes[0].plot(dates_train, y_train_counts, label="Actual", marker="o")
    axes[0].plot(dates_train, y_train_pred, label="Predicted", marker="x")
    axes[0].set_title("Training Period")
    axes[0].set_ylabel("Homeless Count")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if len(X_test) > 0:
        axes[1].plot(dates_test, y_test_counts, label="Actual", marker="o")
        axes[1].plot(dates_test, y_test_pred, label="Predicted", marker="x")
        axes[1].set_title("Testing Period")
        axes[1].set_xlabel("Month")
        axes[1].set_ylabel("Homeless Count")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    else:
        axes[0].set_xlabel("Month")

    fig.suptitle("XGBoost: Actual vs Predicted Homeless Count", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(args.actual_vs_pred_plot, dpi=200)
    plt.close()
    print(f"Saved actual vs predicted plot to {args.actual_vs_pred_plot}")

    feature_importance = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values()
    plt.figure(figsize=(8, 6))
    feature_importance.plot(kind="barh")
    plt.title("XGBoost Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(args.feature_importance_plot, dpi=200)
    plt.close()
    print(f"Saved feature importance plot to {args.feature_importance_plot}")


if __name__ == "__main__":
    main()
