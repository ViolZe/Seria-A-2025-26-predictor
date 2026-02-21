import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Serie A 2025–26 Projection", layout="wide")

st.title("🏆 Serie A 2025–26 Full Table Projection")
st.markdown("Recency-weighted normalized probability model")

# =====================================================
# OFFICIAL 2025–26 TEAM LIST
# =====================================================
official_teams = [
    "[ACM] AC Milan",
    "[ROM] AS Roma",
    "[ATA] Atalanta",
    "[BOL] Bologna",
    "[CAG] Cagliari",
    "[COMO] Como",
    "[GEN] Genoa",
    "[VER] Hellas Verona",
    "[INT] Inter Milan",
    "[JUV] Juventus",
    "[PAR] Parma",
    "[LAZ] Lazio",
    "[NAP] Napoli",
    "[TOR] Torino",
    "[LEC] Lecce",
    "[UDI] Udinese",
    "[FIO] Fiorentina",
    "[CRE] Cremonese",
    "[PISA] Pisa ",
    "[SAS] Sassuolo"
]

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_excel("data/Complete Serie A Dataset Finally.xlsx")
    numeric_cols = ["P", "WIN%", "GD PER GAME", "GA PER GAME"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
    return df

df = load_data()

# =====================================================
# RECENCY CONTROL
# =====================================================
st.sidebar.header("⚙ Model Control")
recency_power = st.sidebar.slider("Recency Weight Strength", 0.5, 3.0, 1.5, 0.1)

# =====================================================
# BUILD RECENCY-WEIGHTED CHAMPION PROFILE
# =====================================================
champions = df[df["CHAMPION"] == 1].copy()
champions["Season_Start"] = champions["YEAR"].str[:4].astype(int)
current_year = champions["Season_Start"].max()

champions["Weight"] = 1 / ((current_year - champions["Season_Start"] + 1) ** recency_power)

metrics = ["WIN%", "GD PER GAME", "GA PER GAME", "P"]
weighted_profile = {}

for metric in metrics:
    weighted_profile[metric] = np.sum(
        champions[metric] * champions["Weight"]
    ) / np.sum(champions["Weight"])

# =====================================================
# PREPARE 2025–26 TEAMS
# =====================================================
latest_season = df["YEAR"].max()
latest_data = df[df["YEAR"] == latest_season].copy()

# Relegation-tier fallback (bottom 3 historical average)
bottom_teams = df.sort_values("POSITION", ascending=False).head(60)
fallback_profile = bottom_teams[metrics].mean()

team_rows = []

for team in official_teams:
    if team in latest_data["Team"].values:
        row = latest_data[latest_data["Team"] == team].iloc[0]
    elif team in df["Team"].values:
        row = df[df["Team"] == team].sort_values("YEAR").iloc[-1]
    else:
        # Pisa fallback
        row = pd.Series({
            "Team": team,
            "WIN%": fallback_profile["WIN%"],
            "GD PER GAME": fallback_profile["GD PER GAME"],
            "GA PER GAME": fallback_profile["GA PER GAME"],
            "P": fallback_profile["P"]
        })
    team_rows.append(row)

teams = pd.DataFrame(team_rows)

std_values = df[metrics].std()

# =====================================================
# DISTANCE CALCULATION
# =====================================================
def calculate_distance(row):
    return np.sqrt(
        ((row["WIN%"] - weighted_profile["WIN%"]) / std_values["WIN%"])**2 +
        ((row["GD PER GAME"] - weighted_profile["GD PER GAME"]) / std_values["GD PER GAME"])**2 +
        ((row["GA PER GAME"] - weighted_profile["GA PER GAME"]) / std_values["GA PER GAME"])**2 +
        ((row["P"] - weighted_profile["P"]) / std_values["P"])**2
    )

teams["Distance"] = teams.apply(calculate_distance, axis=1)
teams["Score"] = 1 / teams["Distance"]
teams["Win Probability"] = (teams["Score"] / teams["Score"].sum()) * 100

# =====================================================
# RANK TABLE
# =====================================================
table = teams[["Team", "Win Probability"]].sort_values(
    "Win Probability", ascending=False
).reset_index(drop=True)

table.index += 1
table.rename_axis("Position", inplace=True)

def competition_label(pos):
    if pos <= 4:
        return "🔵 UCL"
    elif pos == 5:
        return "🟠 Europa"
    elif pos == 6:
        return "🟢 Conference"
    elif pos >= 18:
        return "🔴 Relegation"
    else:
        return ""

table["Competition"] = table.index.map(competition_label)

# =====================================================
# DISPLAY
# =====================================================
st.subheader("📊 Projected 2025–26 League Table")

st.dataframe(
    table.style.format({"Win Probability": "{:.2f}%"}),
    use_container_width=True
)

winner = table.iloc[0]
st.success(f"Predicted 2025–26 Champion: {winner['Team']}")
st.info(f"Win Probability: {winner['Win Probability']:.2f}%")