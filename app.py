# =============================================================================
# app.py — Toyota Corolla Resale Price Predictor (Streamlit UI)
# =============================================================================
# All ML logic lives in model.py. This file is pure UI.
#
# Run locally : streamlit run app.py
# Deploy      : push repo to GitHub → share.streamlit.io → main file: app.py
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from model import load_and_train, predict, sensitivity_analysis

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Corolla Valuation",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ── Plotly helper (Streamlit 1.45+ / legacy compatible) ──────────────────────
def _chart(fig, height=400):
    fig.update_layout(height=height)
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# LOAD & TRAIN  (cached inside model.py)
# =============================================================================
try:
    df, model, features, importances, metrics = load_and_train()
except FileNotFoundError:
    st.error(
        "**ToyotaCorolla.csv not found.**  "
        "Place it in the repo root alongside app.py."
    )
    st.stop()

PRICE_MIN = int(df["Price"].min())
PRICE_MAX = int(df["Price"].max())
PRICE_MED = int(df["Price"].median())


# =============================================================================
# SIDEBAR — vehicle configurator
# =============================================================================
with st.sidebar:
    st.markdown('<p class="sidebar-title">🚗 Vehicle Configurator</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sidebar-caption">Every parameter impacts the estimate</p>',
                unsafe_allow_html=True)

    st.markdown('<p class="sidebar-section">MECHANICAL</p>', unsafe_allow_html=True)
    age    = st.slider("Age (months)",       1,   80,  36,
                       help="Age in months as of August 2004")
    km     = st.slider("Mileage (km)",       0, 250_000, 60_000, step=1_000)
    hp     = st.slider("Horsepower",        60,  192,  90)
    cc     = st.slider("Engine (cc)",     1300, 2000, 1600, step=100)
    weight = st.slider("Weight (kg)",      900, 1615, 1050)
    tax    = st.slider("Quarterly tax (€)", 0,  300,  85)

    st.markdown('<p class="sidebar-section">SAFETY</p>', unsafe_allow_html=True)
    abs_b   = st.toggle("ABS",              value=True)
    airbag1 = st.toggle("Driver airbag",    value=True)
    airbag2 = st.toggle("Passenger airbag", value=True)

    st.markdown('<p class="sidebar-section">COMFORT</p>', unsafe_allow_html=True)
    airco        = st.toggle("Air conditioning",  value=True)
    auto_airco   = st.toggle("Automatic A/C",     value=False)
    pwr_steer    = st.toggle("Power steering",    value=True)
    central_lock = st.toggle("Central lock",      value=True)
    pwr_windows  = st.toggle("Powered windows",   value=False)
    automatic    = st.toggle("Automatic gearbox", value=False)
    boardcomp    = st.toggle("On-board computer", value=False)

    st.markdown('<p class="sidebar-section">EXTRAS</p>', unsafe_allow_html=True)
    met_color = st.toggle("Metallic paint",   value=True)
    sport     = st.toggle("Sport model",      value=False)
    met_rim   = st.toggle("Metallic rims",    value=False)
    cd_player = st.toggle("CD player",        value=False)
    tow_bar   = st.toggle("Tow bar",          value=False)
    mistlamps = st.toggle("Fog lamps",        value=False)
    backseat  = st.toggle("Backseat divider", value=True)

    st.divider()
    st.caption(f"🌲 Random Forest · {metrics['n_trees']} trees · 5-fold CV")


# =============================================================================
# BUILD INPUT VECTOR
# =============================================================================
car = df[features].median().to_dict()
car.update({
    "Age_08_04": age, "KM": km, "HP": hp, "cc": cc,
    "Weight": weight, "Quarterly_Tax": tax,
    "ABS": int(abs_b), "Airbag_1": int(airbag1), "Airbag_2": int(airbag2),
    "Airco": int(airco), "Automatic_airco": int(auto_airco),
    "Power_Steering": int(pwr_steer), "Central_Lock": int(central_lock),
    "Powered_Windows": int(pwr_windows), "Automatic": int(automatic),
    "Boardcomputer": int(boardcomp), "Met_Color": int(met_color),
    "Sport_Model": int(sport), "Metallic_Rim": int(met_rim),
    "CD_Player": int(cd_player), "Tow_Bar": int(tow_bar),
    "Mistlamps": int(mistlamps), "Backseat_Divider": int(backseat),
})

pred, pred_std = predict(model, features, car)
pred_low       = max(PRICE_MIN, pred - pred_std)
pred_high      = min(PRICE_MAX, pred + pred_std)
diff           = pred - PRICE_MED
pct_pos        = (pred - PRICE_MIN) / (PRICE_MAX - PRICE_MIN) * 100
sens           = sensitivity_analysis(model, features, car)


# =============================================================================
# PAGE HEADER
# =============================================================================
st.markdown(
    '<h1 class="page-title">Toyota Corolla — Resale Valuation</h1>'
    '<p class="page-sub">Random Forest · 1,436 vehicles · '
    'Every parameter visibly impacts the estimated price</p>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["💰  Valuation", "📊  Data Explorer", "🌲  Model"])


# =============================================================================
# TAB 1 — VALUATION
# =============================================================================
with tab1:

    hero_col, gauge_col = st.columns([1, 1], gap="large")

    # ── Price hero ────────────────────────────────────────────────────────────
    with hero_col:
        colour = "#34d399" if diff >= 0 else "#f87171"
        sign   = "+" if diff >= 0 else "−"
        st.markdown(f"""
        <div class="price-hero">
            <div class="price-hero-label">ESTIMATED RESALE PRICE</div>
            <div class="price-hero-value">€{pred:,.0f}</div>
            <div class="price-hero-band">
                Confidence range &nbsp;·&nbsp;
                <strong>€{pred_low:,.0f}</strong> – <strong>€{pred_high:,.0f}</strong>
            </div>
            <div class="price-hero-diff" style="color:{colour}">
                {sign}€{abs(diff):,.0f} vs median &nbsp;·&nbsp; {pct_pos:.0f}th percentile
            </div>
        </div>
        """, unsafe_allow_html=True)

        k1, k2, k3 = st.columns(3)
        k1.metric("Test MAE", f"€{metrics['test_mae']:,.0f}")
        k2.metric("R² Score", f"{metrics['test_r2']:.3f}")
        k3.metric("Trees",    str(metrics["n_trees"]))

    # ── Gauge ─────────────────────────────────────────────────────────────────
    with gauge_col:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            number={"prefix": "€", "valueformat": ",.0f",
                    "font": {"size": 32, "color": "#f0f4ff", "family": "Inter"}},
            gauge={
                "axis": {"range": [PRICE_MIN, PRICE_MAX],
                         "tickformat": ",.0f", "tickprefix": "€",
                         "tickfont": {"color": "#4a5568", "size": 9}, "nticks": 6},
                "bar": {"color": "#6366f1", "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                "steps": [
                    {"range": [PRICE_MIN, PRICE_MIN+(PRICE_MAX-PRICE_MIN)*.25], "color": "#0f172a"},
                    {"range": [PRICE_MIN+(PRICE_MAX-PRICE_MIN)*.25,
                               PRICE_MIN+(PRICE_MAX-PRICE_MIN)*.50], "color": "#111827"},
                    {"range": [PRICE_MIN+(PRICE_MAX-PRICE_MIN)*.50,
                               PRICE_MIN+(PRICE_MAX-PRICE_MIN)*.75], "color": "#131e2e"},
                    {"range": [PRICE_MIN+(PRICE_MAX-PRICE_MIN)*.75, PRICE_MAX], "color": "#162036"},
                ],
                "threshold": {"line": {"color": "#fbbf24", "width": 3},
                              "thickness": .8, "value": PRICE_MED},
            },
            title={"text": "Position in market  ·  yellow line = median",
                   "font": {"color": "#64748b", "size": 11}},
        ))
        fig_g.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="#f0f4ff",
            margin=dict(t=50, b=0, l=20, r=20),
        )
        _chart(fig_g, height=270)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Feature impact + sensitivity ─────────────────────────────────────────
    st.markdown('<p class="section-label">FEATURE IMPACT ANALYSIS</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="section-desc">Left: global importance across all 300 trees. '
        'Right: live price swing for a ±10% change on your current configuration.</p>',
        unsafe_allow_html=True,
    )

    imp_col, sens_col = st.columns([3, 2], gap="large")

    with imp_col:
        top15 = importances.head(15).reset_index()
        top15.columns = ["Feature", "Importance"]
        top15["Label"] = top15["Feature"].str.replace("_", " ").str.title()
        top15["Pct"]   = (top15["Importance"] / top15["Importance"].sum() * 100).round(1)

        fig_imp = go.Figure()
        fig_imp.add_trace(go.Bar(
            y=top15["Label"][::-1],
            x=top15["Importance"][::-1],
            orientation="h",
            marker=dict(
                color=top15["Importance"][::-1],
                colorscale=[[0, "#312e81"], [0.5, "#6366f1"], [1, "#a5b4fc"]],
                showscale=False,
            ),
            text=[f"{p}%" for p in top15["Pct"][::-1]],
            textposition="outside",
            textfont=dict(color="#94a3b8", size=10),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        ))
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f0f4ff",
            xaxis=dict(gridcolor="#1e293b", showticklabels=False),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=60, t=10, b=10),
        )
        _chart(fig_imp, height=420)

    with sens_col:
        st.markdown('<p class="section-label" style="margin-top:0">SENSITIVITY — YOUR CONFIG</p>',
                    unsafe_allow_html=True)
        st.markdown(
            '<p class="section-desc">Price swing if parameter increases by 10%</p>',
            unsafe_allow_html=True,
        )
        for feat, swing in sens.items():
            arrow  = "↑" if swing > 0 else "↓"
            colour = "#34d399" if swing > 0 else "#f87171"
            label  = feat.replace("_", " ").title()
            st.markdown(f"""
            <div class="sens-row">
                <span class="sens-label">{label}</span>
                <span class="sens-val" style="color:{colour}">
                    {arrow} €{abs(swing):,.0f}
                </span>
            </div>""", unsafe_allow_html=True)

        st.markdown(
            '<p class="section-desc" style="margin-top:.8rem">'
            '↑ price increases with a 10% rise &nbsp;·&nbsp; ↓ price falls</p>',
            unsafe_allow_html=True,
        )


# =============================================================================
# TAB 2 — DATA EXPLORER
# =============================================================================
with tab2:
    st.markdown('<p class="section-label">SCATTER EXPLORER</p>', unsafe_allow_html=True)

    continuous = [f for f in features if df[f].nunique() > 10]
    c1, c2, c3 = st.columns([1, 1, 2], gap="large")
    with c1:
        x_feat = st.selectbox("X-axis", continuous,
                              index=continuous.index("Age_08_04")
                              if "Age_08_04" in continuous else 0)
    with c2:
        y_feat = st.selectbox("Y-axis", ["Price"] + continuous, index=0)
    with c3:
        show_trend = st.checkbox("Trend line", value=True)
        r = df[[x_feat, y_feat]].corr().iloc[0, 1]
        strength  = "strong" if abs(r) > .5 else ("moderate" if abs(r) > .3 else "weak")
        direction = "positive" if r > 0 else "negative"
        st.markdown(
            f'<span class="stat-badge">Pearson r = {r:.3f} &nbsp;·&nbsp; '
            f'{strength} {direction} correlation</span>',
            unsafe_allow_html=True,
        )

    x_vals = df[x_feat].to_numpy(dtype=float)
    y_vals = df[y_feat].to_numpy(dtype=float)

    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(
        x=x_vals, y=y_vals, mode="markers",
        marker=dict(color=df["Price"].values, colorscale="Viridis",
                    size=5, opacity=.5,
                    colorbar=dict(title="Price (€)",
                                  tickfont=dict(color="#94a3b8"))),
        hovertemplate=f"{x_feat}: %{{x}}<br>{y_feat}: %{{y}}<extra></extra>",
    ))
    if show_trend:
        mask   = np.isfinite(x_vals) & np.isfinite(y_vals)
        coeffs = np.polyfit(x_vals[mask], y_vals[mask], 1)
        xs_t   = np.linspace(x_vals[mask].min(), x_vals[mask].max(), 300)
        fig_sc.add_trace(go.Scatter(
            x=xs_t, y=np.poly1d(coeffs)(xs_t), mode="lines",
            line=dict(color="#fbbf24", width=2, dash="dash"), name="Trend",
        ))
    fig_sc.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f0f4ff",
        xaxis=dict(title=x_feat.replace("_", " "), gridcolor="#1e293b"),
        yaxis=dict(title=y_feat.replace("_", " "), gridcolor="#1e293b"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    _chart(fig_sc, height=420)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">BOX PLOT — PRICE BY BINARY FEATURE</p>',
                unsafe_allow_html=True)

    binary_feats = [c for c in features if df[c].nunique() == 2]
    b1, b2 = st.columns([1, 3], gap="large")
    with b1:
        box_f   = st.selectbox("Group by", binary_feats,
                               index=binary_feats.index("Airco")
                               if "Airco" in binary_feats else 0)
        med_0   = df.loc[df[box_f] == 0, "Price"].median()
        med_1   = df.loc[df[box_f] == 1, "Price"].median()
        premium = med_1 - med_0
        st.markdown(
            f'<span class="stat-badge">Median premium when present: '
            f'{"+" if premium >= 0 else ""}€{premium:,.0f}</span>',
            unsafe_allow_html=True,
        )
    with b2:
        fig_bx = px.box(
            df, x=box_f, y="Price", color=box_f,
            color_discrete_sequence=["#6366f1", "#a5b4fc"],
            points="outliers",
            labels={box_f: box_f.replace("_", " "), "Price": "Price (€)"},
        )
        fig_bx.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f0f4ff", showlegend=False,
            xaxis=dict(gridcolor="#1e293b", tickvals=[0, 1],
                       ticktext=["Absent", "Present"]),
            yaxis=dict(gridcolor="#1e293b"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        _chart(fig_bx, height=360)


# =============================================================================
# TAB 3 — MODEL
# =============================================================================
with tab3:
    st.markdown('<p class="section-label">WHY RANDOM FOREST?</p>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        A single <strong>Decision Tree</strong> always splits first on the most
        dominant features (Age, Mileage). Every other feature only activates
        <em>within</em> those initial buckets — so changing HP or Weight alone
        barely shifts the predicted price. The tool feels unresponsive even when
        the maths is correct.<br><br>
        A <strong>Random Forest</strong> trains 300 trees, each on a <em>random
        subset</em> of features and rows. HP, Weight, and equipment flags each
        become the primary split in a proportion of trees. The ensemble average
        makes every parameter proportionately responsive — essential for a
        valuation tool designed to show how each specification drives value.<br><br>
        The variance across 300 trees also provides a natural
        <strong>confidence band</strong> — something a single tree cannot offer.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-label" style="margin-top:1.5rem">PERFORMANCE</p>',
                unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test MAE",  f"€{metrics['test_mae']:,.0f}")
    m2.metric("Test RMSE", f"€{metrics['test_rmse']:,.0f}")
    m3.metric("R² Score",  f"{metrics['test_r2']:.4f}")
    m4.metric("Trees",     str(metrics["n_trees"]))

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    pa, pb = st.columns(2, gap="large")

    with pa:
        st.markdown('<p class="section-label">PREDICTED vs ACTUAL</p>',
                    unsafe_allow_html=True)
        y_te = metrics["y_test"].values
        y_pr = metrics["y_pred"]
        fig_pva = go.Figure()
        fig_pva.add_trace(go.Scatter(
            x=y_te, y=y_pr, mode="markers",
            marker=dict(color="#6366f1", size=5, opacity=.5),
            hovertemplate="Actual: €%{x:,.0f}<br>Predicted: €%{y:,.0f}<extra></extra>",
        ))
        lm = [PRICE_MIN - 500, PRICE_MAX + 500]
        fig_pva.add_trace(go.Scatter(
            x=lm, y=lm, mode="lines",
            line=dict(color="#fbbf24", dash="dash", width=1.5),
            name="Perfect prediction",
        ))
        fig_pva.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f0f4ff",
            xaxis=dict(title="Actual Price (€)", tickformat=",.0f",
                       gridcolor="#1e293b"),
            yaxis=dict(title="Predicted Price (€)", tickformat=",.0f",
                       gridcolor="#1e293b"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=5, r=5, t=10, b=5),
        )
        _chart(fig_pva, height=380)

    with pb:
        st.markdown('<p class="section-label">RESIDUALS</p>',
                    unsafe_allow_html=True)
        residuals = y_te - y_pr
        fig_res = go.Figure()
        fig_res.add_trace(go.Histogram(
            x=residuals, nbinsx=40,
            marker_color="#6366f1", opacity=.85,
        ))
        fig_res.add_vline(x=0, line_color="#fbbf24", line_dash="dash",
                          line_width=2)
        fig_res.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f0f4ff", showlegend=False,
            xaxis=dict(title="Residual (€) — Actual minus Predicted",
                       tickformat=",.0f", gridcolor="#1e293b"),
            yaxis=dict(title="Count", gridcolor="#1e293b"),
            margin=dict(l=5, r=5, t=10, b=5),
        )
        _chart(fig_res, height=380)
        st.caption(
            f"Mean residual: €{residuals.mean():,.0f} (ideal = 0)  ·  "
            f"Std: €{residuals.std():,.0f}  ·  "
            f"Train MAE: €{metrics['train_mae']:,.0f}  ·  "
            f"Test MAE: €{metrics['test_mae']:,.0f}"
        )
