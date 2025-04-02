import streamlit as st
from model import predict_all_models, get_value_bets
from utils.data_loader import load_fight_card, load_odds

st.set_page_config(page_title="UFC Value Bets", layout="wide")
st.title("🥋 UFC Value Betting Dashboard")

# Load data
fights = load_fight_card()
odds = load_odds()

# Select fight
fight = st.selectbox("Select a Fight", fights)

# Predict
with st.spinner("Predicting..."):
    model_outputs = predict_all_models(fight)

st.subheader("📊 Modelvergelijking: Odds per model")
st.write(model_outputs)

# Gebruik gemiddelde output
model_probs = model_outputs["Average"]

# Value bet check
if fight in odds:
    value_bets = get_value_bets(model_probs, odds[fight])
    st.subheader("💰 Value Bets")
    st.table(value_bets)
else:
    st.warning(f"No odds available for '{fight}'.")
