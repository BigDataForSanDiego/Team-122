# Project Structure

- `README.md` — High-level overview, setup instructions, and usage notes.
- `master.csv` — Base dataset with monthly homeless counts and drivers (temp, rent, unemployment, evictions).
- `linear_model.py` — Baseline linear regression workflow (lags + basic features, CLI-friendly).
- `random_forest_model.py` — RandomForestRegressor pipeline with lag engineering, grid search, metrics, and plots.
- `rf_predictions.csv` — Latest Random Forest prediction export (19-month holdout).
- `rf_actual_vs_pred.png` / `rf_feature_importance.png` — Visualization artifacts from the Random Forest run.
- `predictions/` — Folder of historical linear-model prediction CSVs, each tagged per experiment.
- `scripts/`
  - `add_counts_to_master.py` — Utility to append homelessness counts into `master.csv`.
  - `aggregate_monthly_counts.py` — Aggregates raw data into monthly totals before modeling.
- `mock_actual_vs_predicted.png` — Legacy chart example kept for reference/demo purposes.
