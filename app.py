# app.py

import streamlit as st
from model import train_model, predict

st.set_page_config(page_title="Toyota Corolla Price Predictor")

st.title("🚗 Toyota Corolla Price Predictor")

@st.cache_resource
def load_model():
    model, features, importances, metrics, df = train_model()
    return model, features

model, features = load_model()

st.sidebar.header("Car specifications")

car_input = {}
for f in features:
    car_input[f] = st.sidebar.number_input(f, value=0)

if st.button("Predict price"):
    price, std = predict(model, features, car_input)

    st.metric("Estimated price", f"€{price:,.0f}")
    st.caption(f"Uncertainty ± €{std:,.0f}")