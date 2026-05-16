import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

CSV_PATH = "ToyotaCorolla.csv"
TARGET = "Price"
RANDOM_STATE = 42


def load_data(path: str = CSV_PATH) -> pd.DataFrame:
    """Load the Toyota Corolla dataset and return a clean numeric DataFrame."""
    df = pd.read_csv(path)
    df = df.drop(columns=["Id", "Model", "Fuel_Type"], errors="ignore")
    df = df.select_dtypes(include=[np.number])
    df = df.dropna()
    return df


def train_model(path: str = CSV_PATH):
    """Train a lightweight Random Forest regressor and return model metadata."""
    df = load_data(path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "n_trees": 200,
        "train_mae": float(mean_absolute_error(y_train, model.predict(X_train))),
        "test_mae": float(mean_absolute_error(y_test, y_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "test_r2": float(r2_score(y_test, y_pred)),
        "y_test": y_test,
        "y_pred": y_pred,
    }
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    return model, list(X.columns), importances, metrics, df


def load_and_train(path: str = CSV_PATH):
    """Compatibility wrapper for app.py."""
    model, features, importances, metrics, df = train_model(path)
    return df, model, features, importances, metrics


def predict(model: RandomForestRegressor, features: list, input_dict: dict):
    """Predict with a Random Forest and return mean price + uncertainty."""
    X = pd.DataFrame([{f: input_dict.get(f, 0) for f in features}])[features]
    preds = np.array([tree.predict(X)[0] for tree in model.estimators_])
    return float(preds.mean()), float(preds.std())


def sensitivity_analysis(model, features, car_spec, delta: float = 0.10):
    """Estimate sensitivity for continuous features using ±delta perturbation."""
    continuous = {
        "Age_08_04": car_spec.get("Age_08_04", 36),
        "KM": car_spec.get("KM", 60_000),
        "HP": car_spec.get("HP", 90),
        "Weight": car_spec.get("Weight", 1050),
        "cc": car_spec.get("cc", 1600),
        "Quarterly_Tax": car_spec.get("Quarterly_Tax", 85),
    }
    swings = {}
    for feat, value in continuous.items():
        up_p, _ = predict(model, features, {**car_spec, feat: value * (1 + delta)})
        down_p, _ = predict(model, features, {**car_spec, feat: value * (1 - delta)})
        swings[feat] = up_p - down_p

    return pd.Series(swings).reindex(pd.Series(swings).abs().sort_values(ascending=False).index)


if __name__ == "__main__":
    df, model, features, importances, metrics = load_and_train()
    print("Trained lightweight model")
    print(f"Features: {len(features)}")
    print(f"Test MAE: €{metrics['test_mae']:,.0f}")
    print(f"R²: {metrics['test_r2']:.3f}")
