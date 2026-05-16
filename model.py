# =============================================================================
# model.py — Toyota Corolla Resale Price Predictor
# Full data pipeline: EDA · Normalization · Random Forest · Prediction
# =============================================================================
#
# This file is the single source of truth for all ML logic.
# It is imported by app.py (Streamlit UI) and by the analysis notebook.
#
# Run standalone for a full console analysis + saved plots:
#   python model.py
#
# Inspired by: https://open.hpi.de/courses/datascience2023/overview
# Dataset    : https://www.kaggle.com/datasets/klkwak/toyotacorollacsv
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Try to import Streamlit caching; fall back gracefully when run as script ──
try:
    import streamlit as st
    _cache_data     = st.cache_data
    _cache_resource = st.cache_resource
except ImportError:
    def _cache_data(fn=None, **kw):
        return fn if fn else lambda f: f
    def _cache_resource(fn=None, **kw):
        return fn if fn else lambda f: f


# =============================================================================
# CONFIGURATION
# =============================================================================
CSV_PATH     = "ToyotaCorolla.csv"
TARGET       = "Price"
DROP_COLS    = ["Id", "Model", "Fuel_Type"]
TEST_SIZE    = 0.20
RANDOM_STATE = 42
N_CANDIDATES = [100, 200, 300]      # tree counts to cross-validate
OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120})


# =============================================================================
# 1. DATA LOADING & CLEANING
# =============================================================================

@_cache_data(show_spinner="Loading dataset…")
def load_and_clean(path: str = CSV_PATH) -> pd.DataFrame:
    """
    Load ToyotaCorolla.csv and return a clean, fully numeric DataFrame.

    Steps
    -----
    1. Drop identifier / free-text / categorical columns (Id, Model, Fuel_Type)
    2. Keep only numeric columns
    3. Coerce whole-number floats to int
    4. Drop rows with NaN values

    Why drop Fuel_Type?
    -------------------
    Fuel_Type is categorical (Petrol / Diesel / CNG). Encoding it would add
    complexity without significantly improving the Random Forest (which handles
    the information implicitly through correlated numeric features like HP, cc).
    For a v2 the column could be one-hot encoded.

    Returns
    -------
    pd.DataFrame — clean, integer-typed dataset ready for EDA and modelling
    """
    df_raw = pd.read_csv(path)
    print(f"[load]  Raw shape     : {df_raw.shape[0]} rows × {df_raw.shape[1]} cols")

    df = df_raw.drop(columns=[c for c in DROP_COLS if c in df_raw.columns])
    df = df.select_dtypes(include=[np.number])

    for col in df.columns:
        if df[col].dtype == float:
            if df[col].dropna().apply(lambda x: x == int(x)).all():
                df[col] = df[col].astype("Int64").astype(int)

    before = len(df)
    df.dropna(inplace=True)
    print(f"[clean] Dropped cols  : {DROP_COLS}")
    print(f"[clean] Dropped rows  : {before - len(df)} (NaN)")
    print(f"[clean] Final shape   : {df.shape[0]} rows × {df.shape[1]} cols\n")
    return df.reset_index(drop=True)


# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# =============================================================================

def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics (mean, std, min/max, quartiles,
    range, coefficient of variation) for every column.
    """
    stats = df.describe().T
    stats["range"] = stats["max"] - stats["min"]
    stats["cv"]    = (stats["std"] / stats["mean"]).round(3)
    pd.set_option("display.float_format", "{:,.2f}".format)
    print("=" * 65)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 65)
    print(stats[["count","mean","std","min","25%","50%","75%","max","range","cv"]].to_string())
    print()
    return stats


def correlation_analysis(df: pd.DataFrame) -> pd.Series:
    """
    Compute Pearson correlations of all features with Price,
    sorted by absolute magnitude.
    """
    corr    = df.corr()
    ordered = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False).index
    price_corr = corr[TARGET].drop(TARGET).reindex(ordered)
    print("=" * 65)
    print("CORRELATION WITH PRICE (sorted by |r|)")
    print("=" * 65)
    print(price_corr.to_string())
    print()
    return price_corr


# =============================================================================
# 3. NORMALIZATION
# =============================================================================

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Min-Max normalization → all values scaled to [0, 1].

    Stored as `normalized` to match the pd.DataFrame.normalized convention
    used in the companion notebook.

    Note: Random Forests are scale-invariant; normalization here is for
    visual EDA comparability and for future gradient-based model compatibility.
    """
    scaler     = MinMaxScaler()
    normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    print("=" * 65)
    print("NORMALIZATION — MinMaxScaler → [0, 1]")
    print("=" * 65)
    print(normalized.describe().loc[["min", "max"]].T.to_string())
    print()
    return normalized


# =============================================================================
# 4. VISUALIZATIONS
# =============================================================================

def _save(fig: plt.Figure, name: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[plot]  Saved → {path}")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Lower-triangle heatmap of the full Pearson correlation matrix."""
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.4, ax=ax, annot_kws={"size": 7})
    ax.set_title("Full Correlation Heatmap — Toyota Corolla Dataset",
                 fontsize=14, pad=14)
    plt.tight_layout()
    _save(fig, "01_correlation_heatmap.png")


def plot_top_correlations(price_corr: pd.Series) -> None:
    """Horizontal bar chart of top-12 features correlated with Price."""
    top    = price_corr.abs().sort_values(ascending=False).head(12)
    colors = ["#d7191c" if price_corr[f] < 0 else "#2166ac" for f in top.index]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(top.index[::-1], top.values[::-1],
                   color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Absolute Pearson Correlation with Price")
    ax.set_title("Top 12 Features Correlated with Sale Price", fontsize=13)
    ax.axvline(0.3, color="grey", linestyle="--", linewidth=0.8,
               label="r = 0.3 threshold")
    ax.legend(fontsize=9)
    for bar, feat in zip(bars[::-1], top.index[::-1]):
        direction = "▲" if price_corr[feat] > 0 else "▼"
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                direction, va="center", fontsize=9,
                color="#2166ac" if price_corr[feat] > 0 else "#d7191c")
    plt.tight_layout()
    _save(fig, "02_top_correlations.png")


def plot_scatter_raw(df: pd.DataFrame, price_corr: pd.Series) -> None:
    """Scatter plots (raw values) for the 6 most correlated continuous features."""
    continuous = [f for f in price_corr.abs().sort_values(ascending=False).index
                  if df[f].nunique() > 10][:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, feat in zip(axes.flatten(), continuous):
        r = price_corr[feat]
        ax.scatter(df[feat], df[TARGET], alpha=0.3, s=15,
                   color="steelblue", edgecolors="none")
        z  = np.polyfit(df[feat], df[TARGET], 1)
        xs = np.linspace(df[feat].min(), df[feat].max(), 200)
        ax.plot(xs, np.poly1d(z)(xs), "r--", linewidth=1.5, label=f"r = {r:.2f}")
        ax.set_xlabel(feat, fontsize=10)
        ax.set_ylabel("Price (€)", fontsize=10)
        ax.set_title(f"{feat} vs Price", fontsize=11)
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.suptitle("Scatter Plots: Key Features vs. Sale Price",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, "03_scatter_raw.png")


def plot_scatter_normalized(normalized: pd.DataFrame,
                             price_corr: pd.Series,
                             df: pd.DataFrame) -> None:
    """Scatter plots on the normalized [0,1] scale for slope comparison."""
    continuous = [f for f in price_corr.abs().sort_values(ascending=False).index
                  if df[f].nunique() > 10][:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, feat in zip(axes.flatten(), continuous):
        r = price_corr[feat]
        ax.scatter(normalized[feat], normalized[TARGET], alpha=0.3, s=15,
                   color="darkorange", edgecolors="none")
        z  = np.polyfit(normalized[feat], normalized[TARGET], 1)
        xs = np.linspace(0, 1, 200)
        ax.plot(xs, np.poly1d(z)(xs), "navy", linestyle="--",
                linewidth=1.5, label=f"r = {r:.2f}")
        ax.set_xlabel(f"{feat} (normalized)", fontsize=9)
        ax.set_ylabel("Price (normalized)", fontsize=9)
        ax.set_title(f"{feat} vs Price (norm.)", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    fig.suptitle("Scatter Plots on Normalized Data", fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, "04_scatter_normalized.png")


def plot_boxplots_binary(df: pd.DataFrame, price_corr: pd.Series) -> None:
    """Box plots of Price split by the 12 most correlated binary (0/1) features."""
    binary_cols = [c for c in df.columns
                   if c != TARGET
                   and df[c].nunique() == 2
                   and set(df[c].unique()).issubset({0, 1})]
    top_binary = [f for f in price_corr.abs().sort_values(ascending=False).index
                  if f in binary_cols][:12]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for ax, feat in zip(axes.flatten(), top_binary):
        data = [df.loc[df[feat] == 0, TARGET], df.loc[df[feat] == 1, TARGET]]
        ax.boxplot(data, labels=[f"{feat}=0", f"{feat}=1"],
                   patch_artist=True,
                   boxprops=dict(facecolor="#AED6F1"),
                   medianprops=dict(color="#E74C3C", linewidth=2),
                   whiskerprops=dict(linewidth=1.2),
                   flierprops=dict(marker="o", markersize=3, alpha=0.4))
        ax.set_title(f"{feat}  (r={price_corr[feat]:.2f})", fontsize=9)
        ax.set_ylabel("Price (€)", fontsize=8)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.tick_params(labelsize=8)
    fig.suptitle("Box Plots: Price Distribution by Binary Feature Presence",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, "05_boxplots_binary.png")


def plot_boxplots_age_groups(df: pd.DataFrame) -> None:
    """Box plots of Price binned by car age — illustrates depreciation curve."""
    df_plot = df.copy()
    df_plot["Age_Group"] = pd.cut(
        df_plot["Age_08_04"],
        bins=[0, 12, 24, 36, 48, 60, 120],
        labels=["0–12m", "13–24m", "25–36m", "37–48m", "49–60m", "60m+"],
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    df_plot.boxplot(
        column=TARGET, by="Age_Group", ax=ax, patch_artist=True,
        boxprops=dict(facecolor="#A9DFBF"),
        medianprops=dict(color="#C0392B", linewidth=2),
    )
    ax.set_title("Price Distribution by Car Age Group", fontsize=13)
    ax.set_xlabel("Age Group (months since manufacture)")
    ax.set_ylabel("Price (€)")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.suptitle("")
    plt.tight_layout()
    _save(fig, "06_boxplots_age_groups.png")


# =============================================================================
# 5. RANDOM FOREST MODEL
# =============================================================================

@_cache_resource(show_spinner="Training Random Forest…")
def train(_df: pd.DataFrame):
    """
    Train a Random Forest Regressor with cross-validated n_estimators.

    Why Random Forest over Decision Tree?
    --------------------------------------
    A single Decision Tree always makes its first splits on the most dominant
    features (Age_08_04, KM). All other features only activate *within* those
    initial leaves — so adjusting HP or Weight alone barely moves the price.

    A Random Forest trains N trees, each on a *random subset* of features and
    rows (bootstrap). HP, Weight, safety flags and comfort options each become
    the primary split in a proportion of trees. The ensemble average means every
    parameter contributes visibly and proportionately — essential for a
    valuation tool where the UX goal is to show how each spec drives value.

    Additional benefit: the spread across 300 trees gives a natural confidence
    band — something a single tree cannot provide.

    Parameters
    ----------
    _df : pd.DataFrame — clean dataset (output of load_and_clean)

    Returns
    -------
    (model, features, importances, metrics)
    """
    features = [c for c in _df.columns if c != TARGET]
    X, y     = _df[features], _df[TARGET]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[split] Train : {len(X_tr)}  |  Test : {len(X_te)}  |  Features : {len(features)}\n")

    # Cross-validate to find optimal n_estimators
    best_n, best_mae = N_CANDIDATES[0], float("inf")
    for n in N_CANDIDATES:
        scores = cross_val_score(
            RandomForestRegressor(n_estimators=n, random_state=RANDOM_STATE, n_jobs=-1),
            X_tr, y_tr, cv=5, scoring="neg_mean_absolute_error",
        )
        mae = -scores.mean()
        print(f"[cv]    n={n:>3}  CV MAE = €{mae:,.0f}")
        if mae < best_mae:
            best_mae, best_n = mae, n

    print(f"\n[model] Best n_estimators : {best_n}\n")

    rf = RandomForestRegressor(
        n_estimators=best_n, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)

    y_pred = rf.predict(X_te)
    metrics = dict(
        n_trees   = best_n,
        train_mae = mean_absolute_error(y_tr, rf.predict(X_tr)),
        test_mae  = mean_absolute_error(y_te, y_pred),
        test_rmse = float(np.sqrt(mean_squared_error(y_te, y_pred))),
        test_r2   = float(r2_score(y_te, y_pred)),
        y_test    = y_te,
        y_pred    = y_pred,
    )
    importances = pd.Series(
        rf.feature_importances_, index=features
    ).sort_values(ascending=False)

    return rf, features, importances, metrics


def plot_feature_importance(importances: pd.Series) -> None:
    """Horizontal bar chart of top-15 Random Forest feature importances."""
    fig, ax = plt.subplots(figsize=(10, 6))
    importances.head(15).plot.barh(ax=ax, color="#2980B9", edgecolor="white")
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (mean impurity decrease across 300 trees)")
    ax.set_title("Top 15 Most Important Features — Random Forest", fontsize=12)
    plt.tight_layout()
    _save(fig, "07_feature_importance_rf.png")
    print("Top 10 feature importances:")
    print(importances.head(10).to_string())
    print()


def plot_cv_n_estimators(df: pd.DataFrame) -> None:
    """Plot CV MAE vs n_estimators to justify the chosen tree count."""
    features = [c for c in df.columns if c != TARGET]
    X_tr, _, y_tr, _ = train_test_split(
        df[features], df[TARGET], test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    ns, maes = [], []
    for n in N_CANDIDATES:
        s = cross_val_score(
            RandomForestRegressor(n_estimators=n, random_state=RANDOM_STATE, n_jobs=-1),
            X_tr, y_tr, cv=5, scoring="neg_mean_absolute_error",
        )
        ns.append(n)
        maes.append(-s.mean())
    best_n = ns[int(np.argmin(maes))]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ns, maes, marker="o", color="steelblue", linewidth=2)
    ax.axvline(best_n, color="red", linestyle="--", label=f"Best n = {best_n}")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("CV MAE (€)")
    ax.set_title("Random Forest — Cross-Validated MAE vs Tree Count")
    ax.legend()
    plt.tight_layout()
    _save(fig, "08_cv_n_estimators.png")


def plot_predicted_vs_actual(metrics: dict) -> None:
    """Scatter plot: predicted vs actual prices on the test set."""
    y_te, y_pr = metrics["y_test"].values, metrics["y_pred"]
    lims = [min(y_te.min(), y_pr.min()) - 500, max(y_te.max(), y_pr.max()) + 500]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_te, y_pr, alpha=0.45, s=25, color="steelblue",
               label="Test predictions")
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual Price (€)")
    ax.set_ylabel("Predicted Price (€)")
    ax.set_title("Random Forest: Predicted vs Actual Price (Test Set)", fontsize=12)
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend()
    plt.tight_layout()
    _save(fig, "09_predicted_vs_actual.png")


def plot_residuals(metrics: dict) -> None:
    """Two-panel residual diagnostic: scatter + histogram."""
    y_te, y_pr = metrics["y_test"].values, metrics["y_pred"]
    residuals  = y_te - y_pr
    fig, axes  = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_pr, residuals, alpha=0.45, s=20, color="darkorange")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("Predicted Price (€)")
    axes[0].set_ylabel("Residual (Actual − Predicted)")
    axes[0].set_title("Residuals vs Predicted Price")
    axes[1].hist(residuals, bins=30, color="steelblue", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("Residual (€)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of Residuals")
    for ax in axes:
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.suptitle("Residual Analysis — Random Forest Regressor", fontsize=12)
    plt.tight_layout()
    _save(fig, "10_residuals.png")
    print(f"[residuals] Mean : €{residuals.mean():,.0f}  |  Std : €{residuals.std():,.0f}\n")


# =============================================================================
# 6. PREDICTION UTILITIES  (used by app.py)
# =============================================================================

def load_and_train(path: str = CSV_PATH):
    """
    Public entry point for app.py.
    Returns (df, model, features, importances, metrics).
    """
    df                        = load_and_clean(path)
    rf, features, imp, met    = train(df)
    return df, rf, features, imp, met


def predict(model: RandomForestRegressor,
            features: list,
            car_spec: dict) -> tuple:
    """
    Predict resale price with a confidence band derived from tree variance.

    Parameters
    ----------
    model    : fitted RandomForestRegressor
    features : list of feature names (same order as training)
    car_spec : dict of feature → value (missing keys default to 0)

    Returns
    -------
    (mean_price: float, std_price: float)
        mean_price — average prediction across all trees
        std_price  — std deviation across trees (uncertainty proxy)
    """
    X = pd.DataFrame([{f: car_spec.get(f, 0) for f in features}])[features]
    tree_preds = np.array([t.predict(X)[0] for t in model.estimators_])
    return float(tree_preds.mean()), float(tree_preds.std())


def sensitivity_analysis(model: RandomForestRegressor,
                          features: list,
                          car_spec: dict,
                          delta: float = 0.10) -> pd.Series:
    """
    Compute the price swing for a ±delta relative change on each
    continuous feature, holding all others constant.

    Parameters
    ----------
    delta : float — fractional change (default 0.10 = ±10%)

    Returns
    -------
    pd.Series — price swing (up_price − down_price) per feature, sorted by |swing|
    """
    continuous = {
        "Age_08_04": car_spec.get("Age_08_04", 36),
        "KM":        car_spec.get("KM", 60_000),
        "HP":        car_spec.get("HP", 90),
        "Weight":    car_spec.get("Weight", 1050),
        "cc":        car_spec.get("cc", 1600),
        "Quarterly_Tax": car_spec.get("Quarterly_Tax", 85),
    }
    swings = {}
    for feat, val in continuous.items():
        up_p, _   = predict(model, features, {**car_spec, feat: val * (1 + delta)})
        down_p, _ = predict(model, features, {**car_spec, feat: val * (1 - delta)})
        swings[feat] = up_p - down_p

    return pd.Series(swings).reindex(
        pd.Series(swings).abs().sort_values(ascending=False).index
    )


# =============================================================================
# STANDALONE EXECUTION — full pipeline with console output + saved plots
# =============================================================================

def _evaluate(metrics: dict) -> None:
    print("=" * 65)
    print("MODEL PERFORMANCE — Random Forest")
    print("=" * 65)
    for split in ("train", "test"):
        print(f"  {split.capitalize():<6}  "
              f"MAE = €{metrics[f'{split}_mae']:,.0f}  |  "
              f"RMSE = €{metrics['test_rmse']:,.0f}  |  "
              f"R² = {metrics['test_r2']:.4f}")
    print()


def main() -> None:
    print("\n" + "=" * 65)
    print("  Toyota Corolla Resale Price Analysis")
    print("  Random Forest Pipeline")
    print("=" * 65 + "\n")

    # 1. Load & clean
    df = load_and_clean(CSV_PATH)

    # 2. EDA
    descriptive_stats(df)
    price_corr = correlation_analysis(df)

    # 3. Normalize
    normalized = normalize(df)   # pd.DataFrame.normalized

    # 4. Visualizations
    print("[plots] Generating EDA visualizations …")
    plot_correlation_heatmap(df)
    plot_top_correlations(price_corr)
    plot_scatter_raw(df, price_corr)
    plot_scatter_normalized(normalized, price_corr, df)
    plot_boxplots_binary(df, price_corr)
    plot_boxplots_age_groups(df)

    # 5. Model — CV selection plot (standalone only; train() is cached for app)
    plot_cv_n_estimators(df)
    rf, features, importances, metrics = train(df)
    _evaluate(metrics)
    plot_feature_importance(importances)
    plot_predicted_vs_actual(metrics)
    plot_residuals(metrics)

    # 6. Demo sensitivity
    demo_car = {f: df[f].median() for f in features}
    demo_car.update({"Age_08_04": 36, "KM": 60_000, "HP": 90, "Weight": 1050})
    price, std = predict(rf, features, demo_car)
    sens       = sensitivity_analysis(rf, features, demo_car)

    print("=" * 65)
    print("DEMO PREDICTION")
    print("=" * 65)
    print(f"  Estimated price : €{price:,.0f}  (±€{std:,.0f})")
    print("\nSensitivity (price swing for ±10% change per feature):")
    for feat, swing in sens.items():
        direction = "↑" if swing > 0 else "↓"
        print(f"  {feat:<20} {direction} €{abs(swing):,.0f}")

    print(f"\n[done]  All plots saved to → {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()
