# 🚗 Toyota Corolla — Resale Price Predictor

A machine learning valuation tool for used-car dealerships.  
Built end-to-end in Python: EDA → normalization → Random Forest → deployed Streamlit app.

---

## Live App

👉 **[Open on Streamlit Cloud](https://your-app-url.streamlit.app)**

---

## Repository structure

```
├── app.py                  # Streamlit UI — pure interface layer
├── model.py                # Full pipeline: EDA · normalization · Random Forest · predict
├── style.css               # Design system (dark navy + indigo palette)
├── ToyotaCorolla.csv       # Dataset — 1,436 vehicles, 37 columns
├── requirements.txt        # Python dependencies
└── README.md
```

**`model.py` is the single source of truth.**  
- Imported by `app.py` for the live tool  
- Run standalone (`python model.py`) for a full console analysis + 10 saved plots  

---

## What the app does

| Feature | Description |
|---|---|
| **Instant price estimate** | Adjust any sidebar slider/toggle → price updates live |
| **Confidence band** | ±1 std across 300 trees — surfaces prediction uncertainty |
| **Sensitivity analysis** | Live €-swing of a ±10% change on each mechanical parameter |
| **Feature importance** | Which specs drive value most across the full Random Forest |
| **Data Explorer** | Interactive scatter + box plots with Pearson r annotation |
| **Model tab** | Predicted vs actual, residuals histogram, performance metrics |

---

## Why Random Forest — not a Decision Tree

A single Decision Tree always splits first on the most dominant features (Age, Mileage).  
Every other feature only activates *within* those initial buckets — so changing HP or Weight
alone barely moves the price. The tool feels unresponsive even when the maths is correct.

A **Random Forest** trains 300 trees, each on a *random subset* of features and rows.  
HP, Weight, and equipment flags each become the primary driver in a proportion of trees.  
The ensemble average makes every parameter proportionately responsive — which is essential  
for a valuation interface designed to show *how each specification drives value*.

---

## Run locally

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt

# Launch the web app
streamlit run app.py

# OR run the full analysis pipeline (saves 10 plots to ./outputs/)
python model.py
```

---

## Deploy on Streamlit Community Cloud

1. Push all files to GitHub (repo root — no subfolders)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select repo · branch `main` · main file **`app.py`**
4. Click **Deploy**

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — ToyotaCorolla.csv](https://www.kaggle.com/datasets/klkwak/toyotacorollacsv) |
| Rows | 1,436 vehicles |
| Target | `Price` (resale value in €) |
| Dropped columns | `Id`, `Model`, `Fuel_Type` (non-numeric) |

Inspired by: [OpenHPI — Data Science 2023](https://open.hpi.de/courses/datascience2023/overview)

---

## Known limitations (v1)

| Limitation | Impact | Roadmap fix |
|---|---|---|
| Static dataset (2004) | Prices drift with market | Live data pipeline |
| Confidence band ≈ uncertainty proxy | Not a formal prediction interval | Quantile Regression Forest |
| No regional adjustment | Geography affects prices | Market-zone feature |
| Fuel_Type dropped | Diesel/Petrol premium ignored | One-hot encoding |

---

## Built with

`Python` · `scikit-learn` · `Streamlit` · `Plotly` · `pandas` · `numpy` · `matplotlib` · `seaborn`

---

*Part of the [#BuildInPublic](https://www.linkedin.com/search/results/all/?keywords=%23buildinpublic) series —  
concrete AI applications, shipped and documented honestly.*
