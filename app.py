import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Serie A 2025–26 Predictor", layout="wide")

st.title("🏆 Serie A 2025–26 Champion Predictor")
st.markdown("Recency-weighted trend-based prediction model")


# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_excel("data/Complete Serie A Dataset (1).xlsx")
    numeric_cols = ["P", "WIN%", "GD PER GAME", "GA PER GAME"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
    return df

df = load_data()


# =====================================================
# BUILD CHAMPION PROFILE
# =====================================================
champions = df[df["CHAMPION"] == 1].copy()
champions["Season_Start"] = champions["YEAR"].str[:4].astype(int)
current_year = champions["Season_Start"].max()

champions["Weight"] = 1 / (current_year - champions["Season_Start"] + 1)

metrics = ["WIN%", "GD PER GAME", "GA PER GAME", "P"]

weighted_profile = {}

for metric in metrics:
    weighted_profile[metric] = np.sum(
        champions[metric] * champions["Weight"]
    ) / np.sum(champions["Weight"])


st.subheader("📊 Recency-Weighted Champion Profile")

profile_df = pd.DataFrame(weighted_profile.items(), columns=["Metric", "Value"])
st.dataframe(profile_df)


# =====================================================
# LATEST SEASON DATA
# =====================================================
latest_season = df["YEAR"].max()
current_teams = df[df["YEAR"] == latest_season].copy()


def calculate_distance(row):
    return np.sqrt(
        (row["WIN%"] - weighted_profile["WIN%"])**2 +
        (row["GD PER GAME"] - weighted_profile["GD PER GAME"])**2 +
        (row["GA PER GAME"] - weighted_profile["GA PER GAME"])**2 +
        ((row["P"] - weighted_profile["P"]) / 100)**2
    )


current_teams["Distance"] = current_teams.apply(calculate_distance, axis=1)
current_teams["Score"] = 1 / current_teams["Distance"]
current_teams["Win Probability"] = (
    current_teams["Score"] / current_teams["Score"].sum()
) * 100


prediction = current_teams[
    ["Team", "WIN%", "GD PER GAME", "GA PER GAME", "P", "Win Probability"]
].sort_values("Win Probability", ascending=False)


st.subheader("🏆 2025–26 Champion Prediction")

st.dataframe(
    prediction.style.format({"Win Probability": "{:.2f}%"}),
    use_container_width=True
)


winner = prediction.iloc[0]

st.success(f"Predicted Champion: {winner['Team']}")
st.info(f"Win Probability: {winner['Win Probability']:.2f}%")