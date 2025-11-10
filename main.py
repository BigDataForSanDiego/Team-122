import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/test a simple baseline on monthly data")
    parser.add_argument("--input", default="master_with_counts.csv", help="Input CSV path (default: master_with_counts.csv)")
    parser.add_argument("--test_size", type=int, default=19, help="Holdout size in months (default: 19)")
    parser.add_argument("--output", default="predictions.csv", help="Where to save test predictions")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    # Normalize column names this script expects
    # master_with_counts.csv has: year_month, homeless_count, average_temp, ZORI, unemployment_rate, evictions
    if "average_temp" in df.columns:
        df = df.rename(columns={"average_temp": "avg_temp"})
    if "ZORI" in df.columns:
        df = df.rename(columns={"ZORI": "median_rent_city"})

    # Parse dates and sort
    df["year_month"] = pd.to_datetime(df["year_month"])
    df = df.sort_values("year_month").reset_index(drop=True)

    # Ensure numeric dtypes for used columns
    for col in ["homeless_count", "median_rent_city", "avg_temp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Create lags
    df["homeless_lag1"] = df["homeless_count"].shift(1)
    df["homeless_lag2"] = df["homeless_count"].shift(2)
    df["homeless_lag3"] = df["homeless_count"].shift(3)
    df["rent_lag1"] = df["median_rent_city"].shift(1)
    df["temp_lag1"] = df["avg_temp"].shift(1)

    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "homeless_lag1",
        "homeless_lag2",
        "homeless_lag3",
        "median_rent_city",
        "avg_temp",
        "rent_lag1",
        "temp_lag1",
    ]

    X = df[feature_cols]
    y = df["homeless_count"]

    test_size = args.test_size
    if test_size <= 0 or test_size >= len(df):
        raise ValueError(f"test_size must be between 1 and {len(df)-1}")

    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    # Save predictions with dates for transparency
    out = df.loc[y_test.index, ["year_month"]].copy()
    out["y_true"] = y_test.values
    out["y_pred"] = y_pred
    out.to_csv(args.output, index=False)

    # Print quick metrics
    print(f"RMSE: {rmse:.3f}")
    print(f"R2:   {r2:.3f}")
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
