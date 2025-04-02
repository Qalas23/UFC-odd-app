import streamlit as st
import pandas as pd
from model import predict_method_and_round

st.title("🧠 Method & Round Predictor")

st.write("Simuleer een gevecht en bekijk de voorspelde manier van winnen en ronde.")

df = pd.read_csv("data/fighter_stats.csv")
fighters = df["name"].dropna().unique()

f1 = st.selectbox("🔴 Red Corner", sorted(fighters))
f2 = st.selectbox("🔵 Blue Corner", sorted(fighters), index=1)

if f1 == f2:
    st.warning("Kies twee verschillende vechters.")
else:
    fight_str = f"{f1} vs {f2}"
    method, finish_round = predict_method_and_round(fight_str)

    st.success(f"🔮 Voorspelling: **{fight_str.split(' vs ')[0]} wint via {method} in ronde {finish_round}**")
