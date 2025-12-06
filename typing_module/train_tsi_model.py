# typing_module/train_tsi_model.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
from pathlib import Path


def main():
    this_dir = Path(__file__).resolve().parent
    data_path = this_dir / "tsi_training_data.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {data_path}. "
            "Create typing_module/tsi_training_data.csv with your labeled sessions."
        )

    df = pd.read_csv(data_path)

    feature_cols = [
        "typing_speed_cpm",
        "backspace_rate_per_100",
        "avg_pause_sec",
        "max_pause_sec",
    ]
    X = df[feature_cols]
    y = df["target_tsi"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("R²:", r2_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))

    model_path = this_dir / "tsi_model.pkl"
    joblib.dump(model, model_path)
    print(f"Saved TSI model to {model_path}")


if __name__ == "__main__":
    main()
