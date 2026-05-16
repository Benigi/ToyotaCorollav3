# =============================================================================
# Toyota Corolla — Dealer Pricing Dashboard
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(
    page_title="Dealer Pricing Dashboard",
    page_icon="🚗",
    layout="wide"
)

# =============================================================================
# DATA
# =============================================================================
@st.cache_data
def load_data():
    df = pd.read_csv("ToyotaCorolla.csv")
    df = df.drop(columns=["Id", "Model", "Fuel_Type"], errors="ignore")
    df = df.select_dtypes(include=[np.number])
    df = df.dropna()
    return df

# =============================================================================
# MODEL
# =============================================================================
@st.cache_resource
def train_model(df):
    TARGET = "Price"
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2": r2_score(y_test, preds)
    }

    return model, X.columns.tolist(), metrics

# =============================================================================
# LOAD
# =============================================================================
df = load_data()
model, features, metrics = train_model(df)

PRICE_MED = df["Price"].median()
PRICE_MIN = df["Price"].min()
PRICE_MAX = df["Price"].max()

# =============================================================================
# SIDEBAR INPUT
# =============================================================================
st.sidebar.title("Vehicle Specs")

car = df[features].median().to_dict()

for f in features:
    car[f] = st.sidebar.number_input(f, value=float(car[f]))

# =============================================================================
# PREDICTION + UNCERTAINTY
# =============================================================================
X_input = pd.DataFrame([car])

tree_preds = np.array([tree.predict(X_input)[0] for tree in model.estimators_])

pred = tree_preds.mean()
uncertainty = tree_preds.std()

low_price  = pred - uncertainty
mid_price  = pred
high_price = pred + uncertainty

# =============================================================================
# STRATEGY
# =============================================================================
strategy = st.selectbox(
    "Pricing Strategy",
    ["Fast Sale", "Balanced", "Max Margin"]
)

if strategy == "Fast Sale":
    recommended = low_price
elif strategy == "Balanced":
    recommended = mid_price
else:
    recommended = high_price

# =============================================================================
# HEADER
# =============================================================================
st.title("🚗 Toyota Corolla — Dealer Pricing Dashboard")
st.caption("Intrinsic valuation model · Random Forest · Decision support tool")

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "💰 Pricing Decision",
    "📊 Market Analysis",
    "📈 Strategy",
    "🧠 Model"
])

# =============================================================================
# TAB 1 — PRICING
# =============================================================================
with tab1:

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Price", f"€{pred:,.0f}")
        st.metric("Recommended Price", f"€{recommended:,.0f}")
        st.metric("Uncertainty", f"± €{uncertainty:,.0f}")

        # Deal quality
        z = (pred - PRICE_MED) / uncertainty
        if z < -1:
            st.success("🔥 Undervalued — strong buy opportunity")
        elif z > 1:
            st.error("⚠️ Overpriced — risk of slow sale")
        else:
            st.info("✅ Fairly priced")

    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            gauge={
                "axis": {"range": [PRICE_MIN, PRICE_MAX]},
                "bar": {"color": "lightblue"},
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "value": PRICE_MED
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 2 — MARKET
# =============================================================================
with tab2:

    df["Predicted"] = model.predict(df[features])

    fig = px.scatter(
        df,
        x="Predicted",
        y="Price",
        opacity=0.4
    )

    fig.add_vline(x=pred, line_dash="dash")
    fig.add_hline(y=pred, line_dash="dash")

    st.plotly_chart(fig, use_container_width=True)

    st.caption("Position vs market: below line = undervalued")

# =============================================================================
# TAB 3 — STRATEGY
# =============================================================================
with tab3:

    price_range = np.linspace(pred - 2*uncertainty, pred + 2*uncertainty, 50)

    sale_prob = np.exp(-(price_range - pred) / uncertainty)
    margin = price_range - pred

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_range, y=sale_prob, name="Sale Probability"))
    fig.add_trace(go.Scatter(x=price_range, y=margin, name="Margin"))

    st.plotly_chart(fig, use_container_width=True)

    st.caption("Trade-off between pricing high (margin) and selling fast")

# =============================================================================
# TAB 4 — MODEL
# =============================================================================
with tab4:

    st.metric("MAE", f"€{metrics['MAE']:,.0f}")
    st.metric("RMSE", f"€{metrics['RMSE']:,.0f}")
    st.metric("R²", f"{metrics['R2']:.2f}")

    importances = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values(ascending=False)

    fig = px.bar(importances.head(10))
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"Top driver: {importances.index[0]}")
