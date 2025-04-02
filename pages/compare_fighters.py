import streamlit as st
import pandas as pd
from model import predict_all_models, build_features, get_value_bets

st.set_page_config(page_title="Vergelijk Vechters", layout="wide")
st.title("🥊 Vergelijk Twee Vechters")

st.write("Kies twee vechters om hun statistieken, voorspellingen en mogelijke value bets te bekijken.")

# Load data
df = pd.read_csv("data/fighter_stats.csv")
fighters = df["name"].dropna().unique()

# Dropdowns
f1 = st.selectbox("🔴 Kies vechter 1 (Red Corner)", sorted(fighters))
f2 = st.selectbox("🔵 Kies vechter 2 (Blue Corner)", sorted(fighters), index=1)

if f1 == f2:
    st.warning("⚠️ Kies twee verschillende vechters.")
else:
    fight_str = f"{f1} vs {f2}"
    
    with st.spinner("Voorspellen met modellen..."):
        model_outputs = predict_all_models(fight_str)

    # 🎯 Gemiddelde voorspelling
    avg_red = model_outputs["Average"]["Red Win"]
    avg_blue = model_outputs["Average"]["Blue Win"]

    if avg_red > avg_blue:
        winnaar = f1
        kans = avg_red
    else:
        winnaar = f2
        kans = avg_blue

    st.success(f"💡 Gemiddeld model voorspelt: **{winnaar} wint** met een kans van **{kans:.1%}**")

    # 📊 Tabel met modelresultaten
    st.subheader("📊 Modelvoorspellingen per model")
    st.write(model_outputs)

    # 📈 Bar Chart
    st.subheader("📉 Vergelijk kans per corner (gemiddeld model)")
    st.bar_chart(pd.DataFrame({
        f1: [avg_red],
        f2: [avg_blue]
    }, index=["Winkans"]))

    # 💰 Odds invoeren
    st.subheader("💸 Boekmaker Odds invoeren")
    col1, col2 = st.columns(2)
    with col1:
        odd_red = st.number_input(f"Odd voor {f1}", value=2.0)
    with col2:
        odd_blue = st.number_input(f"Odd voor {f2}", value=2.0)

    # 💡 Value bets
    probs = {
        "Red Win": avg_red,
        "Blue Win": avg_blue
    }
    odds = {
        "Red Win": odd_red,
        "Blue Win": odd_blue
    }

    value_df = get_value_bets(probs, odds)

    st.subheader("✅ Value Bets op basis van odds")
    if value_df.empty:
        st.info("Geen positieve expected value bij deze odds.")
    else:
        st.table(value_df)

    # 📋 Stats vergelijking
    st.subheader("📊 Statistieken vergelijking")
    f1_stats = df[df["name"] == f1].iloc[0]
    f2_stats = df[df["name"] == f2].iloc[0]

    stats_cols = [
        "age", "height", "weight", "reach",
        "SLpM", "SApM", "sig_str_acc", "str_def",
        "td_avg", "td_acc", "td_def", "sub_avg"
    ]

    compare_df = pd.DataFrame({
        "Stat": stats_cols,
        f1: [f1_stats[col] for col in stats_cols],
        f2: [f2_stats[col] for col in stats_cols],
    })

    st.dataframe(compare_df.set_index("Stat"), use_container_width=True)

# 📋 Stat bar comparison
st.subheader("📌 Statistieken Vergelijking (visueel)")

for stat in stats_cols:
    stat1 = f1_stats[stat]
    stat2 = f2_stats[stat]
    max_val = max(stat1, stat2) + 1e-5

    st.markdown(f"**{stat}**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=f1, value=round(stat1, 2))
        st.progress(min(stat1 / max_val, 1.0))
    with col2:
        st.metric(label=f2, value=round(stat2, 2))
        st.progress(min(stat2 / max_val, 1.0))
