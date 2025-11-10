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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


def parse_max_depth(values: List[str]) -> List[Optional[int]]:
    parsed: List[Optional[int]] = []
    for value in values:
        if value.lower() == "none":
            parsed.append(None)
        else:
            parsed.append(int(value))
    return parsed


def to_int_list(values: List[str]) -> List[int]:
    return [int(value) for value in values]


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

    numeric_cols = [
        "homeless_count",
        "median_rent_city",
        "avg_temp",
        "unemployment_rate",
        "evictions",
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

    if len(df) < 104:
        raise ValueError("Need at least 104 months of data before creating lags.")
    df = df.iloc[-104:].reset_index(drop=True)

    df["homeless_count_lag1"] = df["homeless_count"].shift(1)
    df["homeless_count_lag2"] = df["homeless_count"].shift(2)
    df["homeless_count_lag3"] = df["homeless_count"].shift(3)
    df["median_rent_city_lag1"] = df["median_rent_city"].shift(1)
    df["avg_temp_lag1"] = df["avg_temp"].shift(1)
    df["unemployment_rate_lag1"] = df["unemployment_rate"].shift(1)
    df["evictions_lag1"] = df["evictions"].shift(1)

    lag_cols = [
        "homeless_count_lag1",
        "homeless_count_lag2",
        "homeless_count_lag3",
        "median_rent_city_lag1",
        "avg_temp_lag1",
        "unemployment_rate_lag1",
        "evictions_lag1",
    ]
    for col in lag_cols:
        if col in df.columns:
            first_valid = df[col].first_valid_index()
            if first_valid is not None:
                df[col] = df[col].fillna(df.at[first_valid, col])
    df = df.dropna().reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/test RandomForestRegressor on homelessness data.")
    parser.add_argument("--input", default="master.csv", help="Input CSV path (default: master.csv)")
    parser.add_argument("--output", default="rf_predictions.csv", help="Where to save Random Forest predictions")
    parser.add_argument(
        "--actual-vs-pred-plot",
        default="rf_actual_vs_pred.png",
        help="Path to save actual vs predicted plot",
    )
    parser.add_argument(
        "--feature-importance-plot",
        default="rf_feature_importance.png",
        help="Path to save feature importance plot",
    )
    parser.add_argument(
        "--n-estimators",
        nargs="+",
        default=["200", "400", "600"],
        help="Candidate n_estimators values for tuning (default: 200 400 600)",
    )
    parser.add_argument(
        "--max-depth",
        nargs="+",
        default=["10", "20", "None"],
        help='Candidate max_depth values for tuning (use "None" for unlimited)',
    )
    parser.add_argument(
        "--random-state",
        nargs="+",
        default=["13", "42", "87"],
        help="Candidate random_state values for tuning",
    )
    args = parser.parse_args()

    df = prepare_dataframe(Path(args.input))

    feature_cols = [
        "homeless_count_lag1",
        "homeless_count_lag2",
        "homeless_count_lag3",
        "median_rent_city",
        "avg_temp",
        "unemployment_rate",
        "evictions",
        "median_rent_city_lag1",
        "avg_temp_lag1",
        "unemployment_rate_lag1",
        "evictions_lag1",
    ]
    X = df[feature_cols]
    y = df["homeless_count"]
    dates = df["year_month"]

    train_size = 85
    if len(df) != 104:
        raise ValueError("Prepared dataset must contain exactly 104 months to match the specified split.")
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    dates_test = dates.iloc[train_size:]

    param_grid = {
        "n_estimators": to_int_list(args.n_estimators),
        "max_depth": parse_max_depth(args.max_depth),
        "random_state": to_int_list(args.random_state),
    }
    rf = RandomForestRegressor(n_jobs=1)
    tscv = TimeSeriesSplit(n_splits=3)
    grid = GridSearchCV(
        rf,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    best_model: RandomForestRegressor = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    out_df = pd.DataFrame(
        {
            "year_month": dates_test.dt.strftime("%Y-%m-%d"),
            "y_true": y_test.values,
            "y_pred": y_pred,
        }
    )
    out_df.to_csv(args.output, index=False)

    print("Best Params:", grid.best_params_)
    print(f"RMSE: {rmse:.3f}")
    print(f"R2:   {r2:.3f}")
    print(f"Saved predictions to {args.output}")

    plt.figure(figsize=(10, 5))
    plt.plot(dates_test, y_test, label="Actual", marker="o")
    plt.plot(dates_test, y_pred, label="Predicted", marker="x")
    plt.title("Random Forest: Actual vs Predicted Homeless Count")
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
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(args.feature_importance_plot, dpi=200)
    plt.close()
    print(f"Saved feature importance plot to {args.feature_importance_plot}")


if __name__ == "__main__":
    main()
