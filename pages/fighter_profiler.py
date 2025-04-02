import streamlit as st
import pandas as pd

st.title("🧬 Fighter Profiler")

st.write("Bekijk alle stats van een specifieke vechter.")

df = pd.read_csv("data/fighter_stats.csv")
fighters = df["name"].dropna().unique()
fighter = st.selectbox("Kies een vechter", sorted(fighters))

data = df[df["name"] == fighter].iloc[0]
st.header(f"📈 Statistieken voor {fighter}")

for stat in data.index:
    if stat == "name": continue
    value = data[stat]
    if isinstance(value, (int, float)):
        st.write(f"**{stat}**: {round(value, 2)}")
    else:
        st.write(f"**{stat}**: {value}")


# (optioneel: fight history tonen)
try:
    history = pd.read_csv("data/fight_history.csv")
    fighter_fights = history[(history["fighter"] == fighter)]
    st.subheader("🗓️ Fight History")
    st.table(fighter_fights)
except:
    st.info("Geen fight history CSV gevonden (optioneel)")
