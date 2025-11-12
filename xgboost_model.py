import argparse
import os
from pathlib import Path
from typing import List, Optional

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


def to_int_list(values: List[str]) -> List[int]:
    return [int(value) for value in values]


def to_float_list(values: List[str]) -> List[float]:
    return [float(value) for value in values]


def prepare_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "average_temp" in df.columns:
        df = df.rename(columns={"average_temp": "avg_temp"})
    if "ZORI" in df.columns:
        df = df.rename(columns={"ZORI": "median_rent_city"})

    df["year_month"] = pd.to_datetime(df["year_month"])
    df = df.sort_values("year_month").reset_index(drop=True)
    if "homeless_count" not in df.columns:
        raise ValueError("Input data must include homeless_count column.")
    full_range = pd.date_range(df["year_month"].min(), df["year_month"].max(), freq="MS")
    df = (
        df.set_index("year_month")
        .reindex(full_range)
        .rename_axis("year_month")
        .reset_index()
    )

    numeric_cols = [
        "homeless_count",
        "median_rent_city",
        "avg_temp",
        "unemployment_rate",
        "evictions",
        "precipitation",
        "shelter_beds",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    required_for_span = ["homeless_count", "avg_temp", "median_rent_city", "unemployment_rate", "evictions"]
    for col in required_for_span:
        if col not in df.columns:
            raise ValueError(f"Input data missing required column: {col}")
    completeness_mask = df[required_for_span].notna().all(axis=1)
    if not completeness_mask.any():
        raise ValueError("Unable to find any rows with complete numeric data.")
    last_valid_date = df.loc[completeness_mask, "year_month"].max()
    df = df[df["year_month"] <= last_valid_date].reset_index(drop=True)

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear", limit_direction="both").ffill().bfill()

    covid_present = "covid" in df.columns
    if covid_present:
        df["covid"] = pd.to_numeric(df["covid"], errors="coerce")
        df["covid"] = df["covid"].ffill().bfill().fillna(0)
        df["covid_flag_current"] = (df["covid"] >= 0.5).astype(int)
        df["covid_flag_lag1"] = df["covid_flag_current"].shift(1)
        df["covid_flag_lag3"] = df["covid_flag_current"].shift(3)
        df["covid_flag_roll3"] = df["covid_flag_current"].rolling(window=3, min_periods=1).mean().shift(1)

    if len(df) < 104:
        raise ValueError("Need at least 104 months of data before creating lags.")
    df = df.iloc[-104:].reset_index(drop=True)
    df = df.drop(columns=[col for col in ["cpi", "industrial_production"] if col in df.columns])

    # Homeless count lags 1-6
    homeless_lag_cols = []
    for lag in range(1, 7):
        col = f"homeless_count_lag{lag}"
        df[col] = df["homeless_count"].shift(lag)
        homeless_lag_cols.append(col)

    driver_lag_strategy = {
        "median_rent_city": (1, 2),
        "unemployment_rate": (1, 2),
        "evictions": (1, 2),
    }
    driver_lag_cols = []
    for driver, lags in driver_lag_strategy.items():
        if driver in df.columns:
            for lag in lags:
                col = f"{driver}_lag{lag}"
                df[col] = df[driver].shift(lag)
                driver_lag_cols.append(col)

    lag_cols = homeless_lag_cols + driver_lag_cols
    if "covid_flag_lag1" in df.columns:
        lag_cols.extend(["covid_flag_lag1", "covid_flag_lag3", "covid_flag_roll3"])
    for col in lag_cols:
        if col in df.columns:
            first_valid = df[col].first_valid_index()
            if first_valid is not None:
                df[col] = df[col].fillna(df.at[first_valid, col])
    df = df.dropna().reset_index(drop=True)
    df["month_index"] = range(len(df))
    df["month"] = df["year_month"].dt.month
    df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)
    df["is_winter"] = df["month"].isin([12, 1, 2]).astype(int)
    df["is_fall"] = df["month"].isin([9, 10, 11]).astype(int)
    df["is_spring"] = df["month"].isin([3, 4, 5]).astype(int)
    df["pandemic_period"] = ((df["year_month"] >= "2020-03-01") & (df["year_month"] <= "2022-06-01")).astype(int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/test XGBoost regression on homelessness data.")
    parser.add_argument("--input", default="with_covid_flag.csv", help="Input CSV path (default: with_covid_flag.csv)")
    parser.add_argument("--output", default="xgb_predictions.csv", help="Where to save XGBoost predictions")
    parser.add_argument("--train-output", default="xgb_predictions_train.csv", help="Where to save training predictions")
    parser.add_argument(
        "--actual-vs-pred-plot",
        default="xgb_actual_vs_pred.png",
        help="Path to save actual vs predicted plot",
    )
    parser.add_argument(
        "--feature-importance-plot",
        default="xgb_feature_importance.png",
        help="Path to save feature importance plot",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=75,
        help="Number of boosting rounds (default: 75)",
    )
    parser.add_argument("--max-depth", type=int, default=3, help="Tree depth (default: 3)")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate (default: 0.05)")
    parser.add_argument("--min-child-weight", type=int, default=4, help="min_child_weight (default: 4)")
    args = parser.parse_args()

    df = prepare_dataframe(Path(args.input))

    feature_cols = [
        "median_rent_city",
        "avg_temp",
        "unemployment_rate",
        "evictions",
        "precipitation",
        "shelter_beds",
        "month_index",
        "is_summer",
        "is_winter",
        "is_fall",
        "is_spring",
        "pandemic_period",
    ]
    feature_cols += [f"homeless_count_lag{lag}" for lag in range(1, 7)]
    for driver, lags in [
        ("median_rent_city", (1, 2)),
        ("unemployment_rate", (1, 2)),
        ("evictions", (1, 2)),
    ]:
        for lag in lags:
            feature_cols.append(f"{driver}_lag{lag}")
    if "covid_flag_current" in df.columns:
        feature_cols.extend(
            [
                "covid_flag_current",
                "covid_flag_lag1",
                "covid_flag_lag3",
                "covid_flag_roll3",
            ]
        )

    X = df[feature_cols]
    y = df["homeless_count"]
    dates = df["year_month"]

    test_size = 14
    if len(df) != 104:
        raise ValueError("Prepared dataset must contain exactly 104 months to match the specified split.")
    train_size = len(df) - test_size
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    dates_test = dates.iloc[train_size:]

    param_grid = {
        "reg_lambda": [0.0, 1.0],
        "reg_alpha": [0.0, 0.5],
    }
    xgb = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_jobs=1,
        random_state=13,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_child_weight=args.min_child_weight,
        subsample=0.9,
        colsample_bytree=0.9,
    )
    tscv = TimeSeriesSplit(n_splits=3)
    grid = GridSearchCV(
        xgb,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    best_model: XGBRegressor = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    y_pred_train = best_model.predict(X_train)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    bias = (y_pred - y_test).mean()
    train_rmse = mean_squared_error(y_train, y_pred_train) ** 0.5
    train_r2 = r2_score(y_train, y_pred_train)
    train_bias = (y_pred_train - y_train).mean()

    out_df = pd.DataFrame(
        {
            "year_month": dates_test.dt.strftime("%Y-%m-%d"),
            "y_true": y_test.values,
            "y_pred": y_pred,
        }
    )
    out_df.to_csv(args.output, index=False)
    train_df = pd.DataFrame(
        {
            "year_month": dates.iloc[:train_size].dt.strftime("%Y-%m-%d"),
            "y_true": y_train.values,
            "y_pred": y_pred_train,
        }
    )
    train_df.to_csv(args.train_output, index=False)

    print("Best Params:", grid.best_params_)
    print(f"RMSE: {rmse:.3f}")
    print(f"R2:   {r2:.3f}")
    print(f"Saved predictions to {args.output}")
    print(f"Bias (mean y_pred - y_true): {bias:.3f}")
    print(f"Training RMSE: {train_rmse:.3f}")
    print(f"Training R2:   {train_r2:.3f}")
    print(f"Training Bias: {train_bias:.3f}")

    plt.figure(figsize=(10, 5))
    plt.plot(dates_test, y_test, label="Actual", marker="o")
    plt.plot(dates_test, y_pred, label="Predicted", marker="x")
    plt.title("XGBoost: Actual vs Predicted Homeless Count")
    plt.xlabel("Month")
    plt.ylabel("Homeless Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(args.actual_vs_pred_plot, dpi=200)
    plt.close()
    print(f"Saved actual vs predicted plot to {args.actual_vs_pred_plot}")

    feature_importance = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values()
    plt.figure(figsize=(8, 6))
    feature_importance.plot(kind="barh")
    plt.title("XGBoost Feature Importance")
    plt.xlabel("Gain")
    plt.tight_layout()
    plt.savefig(args.feature_importance_plot, dpi=200)
    plt.close()
    print(f"Saved feature importance plot to {args.feature_importance_plot}")


if __name__ == "__main__":
    main()
