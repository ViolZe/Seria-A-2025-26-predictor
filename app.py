import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Serie A Champion Predictor", layout="wide")

st.title("🏆 Advanced Serie A Champion Predictor")
st.markdown("Recency-weighted + fully normalized similarity model")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_excel("data\Complete Serie A Dataset Final.xlsx")
    numeric_cols = ["P", "WIN%", "GD PER GAME", "GA PER GAME"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
    return df

df = load_data()

# =====================================================
# SIDEBAR CONTROLS
# =====================================================
st.sidebar.header("⚙ Model Controls")

season_list = sorted(df["YEAR"].unique())
selected_season = st.sidebar.selectbox("Select Season to Predict Next Winner", season_list)

recency_power = st.sidebar.slider("Recency Weight Strength", 0.5, 3.0, 1.5, 0.1)

st.sidebar.subheader("Metric Weights")

w_win = st.sidebar.slider("Weight: WIN%", 0.0, 3.0, 1.0, 0.1)
w_gd  = st.sidebar.slider("Weight: GD per Game", 0.0, 3.0, 1.0, 0.1)
w_ga  = st.sidebar.slider("Weight: GA per Game", 0.0, 3.0, 1.0, 0.1)
w_pts = st.sidebar.slider("Weight: Points", 0.0, 3.0, 1.0, 0.1)

# =====================================================
# BUILD RECENCY-WEIGHTED CHAMPION PROFILE
# =====================================================
champions = df[df["CHAMPION"] == 1].copy()
champions["Season_Start"] = champions["YEAR"].str[:4].astype(int)
current_year = champions["Season_Start"].max()

# Advanced recency weighting
champions["Weight"] = 1 / ((current_year - champions["Season_Start"] + 1) ** recency_power)

metrics = ["WIN%", "GD PER GAME", "GA PER GAME", "P"]

weighted_profile = {}

for metric in metrics:
    weighted_profile[metric] = np.sum(
        champions[metric] * champions["Weight"]
    ) / np.sum(champions["Weight"])

st.subheader("📊 Recency-Weighted Champion Profile")

profile_df = pd.DataFrame(weighted_profile.items(), columns=["Metric", "Value"])
st.dataframe(profile_df.style.format({"Value": "{:.3f}"}))

# =====================================================
# GET SELECTED SEASON TEAMS
# =====================================================
current_teams = df[df["YEAR"] == selected_season].copy()

# Standardize metrics
std_values = df[metrics].std()

# =====================================================
# DISTANCE FUNCTION (FULLY NORMALIZED + WEIGHTED)
# =====================================================
def calculate_distance(row):
    return np.sqrt(
        w_win * ((row["WIN%"] - weighted_profile["WIN%"]) / std_values["WIN%"])**2 +
        w_gd  * ((row["GD PER GAME"] - weighted_profile["GD PER GAME"]) / std_values["GD PER GAME"])**2 +
        w_ga  * ((row["GA PER GAME"] - weighted_profile["GA PER GAME"]) / std_values["GA PER GAME"])**2 +
        w_pts * ((row["P"] - weighted_profile["P"]) / std_values["P"])**2
    )

current_teams["Distance"] = current_teams.apply(calculate_distance, axis=1)

# Convert distance → score → probability
current_teams["Score"] = 1 / current_teams["Distance"]
current_teams["Win Probability"] = (
    current_teams["Score"] / current_teams["Score"].sum()
) * 100

prediction = current_teams[
    ["Team", "WIN%", "GD PER GAME", "GA PER GAME", "P", "Win Probability"]
].sort_values("Win Probability", ascending=False)

# =====================================================
# DISPLAY RESULTS
# =====================================================
st.subheader(f"🏆 Predicted Winner for Next Season (Based on {selected_season})")

st.dataframe(
    prediction.style.format({
        "WIN%": "{:.2%}",
        "Win Probability": "{:.2f}%"
    }),
    use_container_width=True
)

winner = prediction.iloc[0]

st.success(f"Predicted Champion: {winner['Team']}")
st.info(f"Win Probability: {winner['Win Probability']:.2f}%")

# =====================================================
# VISUALIZATION
# =====================================================
st.subheader("📈 Win Probability Chart")

fig, ax = plt.subplots()
ax.barh(prediction["Team"], prediction["Win Probability"])
ax.invert_yaxis()
ax.set_xlabel("Win Probability (%)")
st.pyplot(fig)