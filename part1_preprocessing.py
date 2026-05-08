"""
=============================================================================
PART 1: PREPROCESSING PIPELINE
=============================================================================
Project : Attention-Based Temporal Deep Learning for NBA Shot Prediction
HOW TO RUN:  python part1_preprocessing.py
Place CSVs in ./data/  →  output goes to ./processed_data/
DEPENDENCIES: pip install pandas numpy scikit-learn matplotlib seaborn
=============================================================================
"""
import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_DIR   = "."
OUTPUT_DIR = "./processed_data"
SEQ_LEN    = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)

print("=" * 65)
print("  NBA SHOT PREDICTION — PART 1: PREPROCESSING PIPELINE")
print("=" * 65)

# ── STEP 1: LOAD ──────────────────────────────────────────────
print("\n[STEP 1] Loading datasets …")
shot_logs     = pd.read_csv(f"{DATA_DIR}/shot_logs.csv")
games         = pd.read_csv(f"{DATA_DIR}/games.csv")
games_details = pd.read_csv(f"{DATA_DIR}/games_details.csv", low_memory=False)
ranking       = pd.read_csv(f"{DATA_DIR}/ranking.csv")
players       = pd.read_csv(f"{DATA_DIR}/players.csv")
teams         = pd.read_csv(f"{DATA_DIR}/teams.csv")
for name, ds in [("shot_logs", shot_logs),("games", games),("games_details", games_details),
                 ("ranking", ranking),("players", players),("teams", teams)]:
    print(f"  {name:<14}: {ds.shape[0]:>7,} rows × {ds.shape[1]} cols")

# ── STEP 2: CLEAN SHOT_LOGS ───────────────────────────────────
print("\n[STEP 2] Cleaning shot_logs …")
df = shot_logs.copy()
df.drop_duplicates(inplace=True)
df.columns = [c.upper().strip() for c in df.columns]

df["TARGET"] = (df["SHOT_RESULT"].str.lower().str.strip() == "made").astype(int)

def clock_to_sec(s):
    try:
        m, sec = str(s).split(":")
        return int(m) * 60 + int(sec)
    except Exception:
        return np.nan

df["GAME_CLOCK_SEC"] = df["GAME_CLOCK"].apply(clock_to_sec)
global_sc_med = df["SHOT_CLOCK"].median()
df["SHOT_CLOCK"] = df.groupby("PLAYER_ID")["SHOT_CLOCK"].transform(lambda x: x.fillna(x.median()))
df["SHOT_CLOCK"].fillna(global_sc_med, inplace=True)
df["LOCATION_ENC"]     = (df["LOCATION"] == "H").astype(int)
df["WIN_ENC"]          = (df["W"] == "W").astype(int)
df["PTS_TYPE"]         = df["PTS_TYPE"].astype(int)
df["FINAL_MARGIN_ABS"] = df["FINAL_MARGIN"].abs()
df["GAME_COMPETITIVE"] = (df["FINAL_MARGIN_ABS"] <= 5).astype(int)
print(f"  Rows: {df.shape[0]:,}  |  Made={df['TARGET'].sum():,}  Missed={(1-df['TARGET']).sum():,}")

# ── STEP 3: MERGE GAMES ───────────────────────────────────────
print("\n[STEP 3] Merging game context …")
g = games[["GAME_ID","GAME_DATE_EST","HOME_TEAM_ID","VISITOR_TEAM_ID","HOME_TEAM_WINS"]].copy()
g.rename(columns={"GAME_DATE_EST": "GAME_DATE"}, inplace=True)
g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"])
df = df.merge(g, on="GAME_ID", how="left")
print(f"  Shape: {df.shape}")

# ── STEP 4: MERGE GAMES_DETAILS ──────────────────────────────
print("\n[STEP 4] Merging player game-level stats …")
keep_gd = ["GAME_ID","PLAYER_ID","MIN","FG_PCT","FG3_PCT","REB","AST","TO","PTS","PLUS_MINUS"]
gd = games_details[keep_gd].copy()
def min_float(m):
    try:
        p = str(m).split(":"); return float(p[0]) + float(p[1])/60
    except: return np.nan
gd["MIN"] = gd["MIN"].apply(min_float)
gd.rename(columns={"MIN":"MINUTES","FG_PCT":"GD_FG_PCT","FG3_PCT":"GD_FG3_PCT",
                    "REB":"GD_REB","AST":"GD_AST","TO":"GD_TO","PTS":"GD_PTS",
                    "PLUS_MINUS":"GD_PLUS_MINUS"}, inplace=True)
num_gd = ["MINUTES","GD_FG_PCT","GD_FG3_PCT","GD_REB","GD_AST","GD_TO","GD_PTS","GD_PLUS_MINUS"]
gd[num_gd] = gd[num_gd].fillna(0)
df = df.merge(gd, on=["GAME_ID","PLAYER_ID"], how="left")
print(f"  Shape: {df.shape}")

# ── STEP 5: MERGE RANKING ─────────────────────────────────────
print("\n[STEP 5] Merging team ranking …")
rank_season = ranking[ranking["SEASON_ID"] == 22014][["TEAM_ID","W_PCT"]].copy()
rank_avg    = rank_season.groupby("TEAM_ID")["W_PCT"].mean().reset_index()
rank_avg.columns = ["TEAM_ID","TEAM_W_PCT"]
player_team = players[["PLAYER_ID","TEAM_ID"]].drop_duplicates("PLAYER_ID").copy()
df = df.merge(player_team, on="PLAYER_ID", how="left")
df = df.merge(rank_avg, on="TEAM_ID", how="left")
df["TEAM_W_PCT"].fillna(df["TEAM_W_PCT"].median(), inplace=True)
df["OPP_TEAM_ID"] = np.where(df["LOCATION"]=="H", df["VISITOR_TEAM_ID"], df["HOME_TEAM_ID"])
rank_opp = rank_avg.rename(columns={"TEAM_ID":"OPP_TEAM_ID","TEAM_W_PCT":"OPP_W_PCT"})
df = df.merge(rank_opp, on="OPP_TEAM_ID", how="left")
df["OPP_W_PCT"].fillna(df["OPP_W_PCT"].median(), inplace=True)
print(f"  Shape: {df.shape}")

# ── STEP 6: TEMPORAL SORT ────────────────────────────────────
print("\n[STEP 6] Sorting temporally …")
df.sort_values(["PLAYER_ID","GAME_DATE","GAME_ID","SHOT_NUMBER"], ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)

# ── STEP 7: ROLLING FEATURES (transform — no column loss) ────
print("\n[STEP 7] Engineering rolling / momentum features …")
def roll(series, w): return series.shift(1).rolling(w, min_periods=1).mean()

grp = df.groupby("PLAYER_ID")
df["ROLL_FG_5"]   = grp["FGM"].transform(lambda x: roll(x, 5))
df["ROLL_FG_10"]  = grp["FGM"].transform(lambda x: roll(x, 10))
df["ROLL_DIST_5"] = grp["SHOT_DIST"].transform(lambda x: roll(x, 5))
df["ROLL_DEF_5"]  = grp["CLOSE_DEF_DIST"].transform(lambda x: roll(x, 5))
df["ROLL_SC_5"]   = grp["SHOT_CLOCK"].transform(lambda x: roll(x, 5))
df["ROLL_PTS_5"]  = grp["PTS"].transform(lambda x: roll(x, 5))

def made_streak(fg_series):
    fg = fg_series.shift(1).fillna(0).astype(int).values
    streak = np.zeros(len(fg), dtype=int)
    for i in range(1, len(fg)):
        streak[i] = streak[i-1] + 1 if fg[i] == 1 else 0
    return pd.Series(streak, index=fg_series.index)

df["MADE_STREAK"]          = grp["FGM"].transform(made_streak)
df["SHOTS_IN_GAME_SO_FAR"] = df.groupby(["PLAYER_ID","GAME_ID"]).cumcount()

roll_cols = ["ROLL_FG_5","ROLL_FG_10","ROLL_DIST_5","ROLL_DEF_5",
             "ROLL_SC_5","ROLL_PTS_5","MADE_STREAK","SHOTS_IN_GAME_SO_FAR"]
for col in roll_cols:
    df[col].fillna(df[col].median(), inplace=True)
print("  Done.")

# ── STEP 8: FEATURE SELECTION ─────────────────────────────────
print("\n[STEP 8] Selecting features …")
SEQUENCE_FEATURES = [
    "SHOT_DIST","CLOSE_DEF_DIST","SHOT_CLOCK","TOUCH_TIME","DRIBBLES","PTS_TYPE",
    "PERIOD","GAME_CLOCK_SEC","FINAL_MARGIN","LOCATION_ENC","GAME_COMPETITIVE",
    "MINUTES","GD_FG_PCT","GD_FG3_PCT","GD_PTS","GD_AST","GD_REB","GD_PLUS_MINUS","GD_TO",
    "TEAM_W_PCT","OPP_W_PCT",
    "ROLL_FG_5","ROLL_FG_10","ROLL_DIST_5","ROLL_DEF_5","ROLL_SC_5","ROLL_PTS_5",
    "MADE_STREAK","SHOTS_IN_GAME_SO_FAR",
]
print(f"  Features: {len(SEQUENCE_FEATURES)}")
META = ["TARGET","PLAYER_ID","GAME_ID","SHOT_NUMBER","GAME_DATE"]
df_model = df[SEQUENCE_FEATURES + META].copy()
df_model.dropna(subset=SEQUENCE_FEATURES, inplace=True)
df_model.reset_index(drop=True, inplace=True)
print(f"  Rows after dropna: {df_model.shape[0]:,}")

# ── STEP 9: SCALE ─────────────────────────────────────────────
print("\n[STEP 9] Scaling …")
scaler = StandardScaler()
df_model[SEQUENCE_FEATURES] = scaler.fit_transform(df_model[SEQUENCE_FEATURES])

# ── STEP 10: GENERATE SEQUENCES ───────────────────────────────
print(f"\n[STEP 10] Generating sequences (len={SEQ_LEN}) …")
X_seqs, y_labs = [], []
for pid, grp_df in df_model.groupby("PLAYER_ID"):
    grp_df = grp_df.sort_values(["GAME_DATE","GAME_ID","SHOT_NUMBER"])
    feat   = grp_df[SEQUENCE_FEATURES].values
    tgt    = grp_df["TARGET"].values
    for i in range(SEQ_LEN, len(grp_df)):
        X_seqs.append(feat[i-SEQ_LEN:i])
        y_labs.append(tgt[i])

X_sequences = np.array(X_seqs, dtype=np.float32)
y_labels    = np.array(y_labs, dtype=np.int32)
X_flat      = X_sequences.reshape(len(X_sequences), -1)
print(f"  X_sequences : {X_sequences.shape}")
print(f"  y_labels    : {y_labels.shape}")
print(f"  X_flat      : {X_flat.shape}")
print(f"  Made={y_labels.sum():,} ({y_labels.mean()*100:.1f}%)  Missed={(1-y_labels).sum():,}")

# ── STEP 11: EDA PLOTS ────────────────────────────────────────
print("\n[STEP 11] Generating EDA plots …")
plt.style.use("seaborn-v0_8-whitegrid")

# Figure 1: Shot distributions
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("NBA Shot Logs — EDA", fontsize=14, fontweight="bold")
counts = [int(y_labels.sum()), int((1-y_labels).sum())]
bars = axes[0].bar(["Made","Missed"], counts, color=["#2ecc71","#e74c3c"], edgecolor="k", lw=0.8)
axes[0].set_title("Shot Result Distribution", fontweight="bold"); axes[0].set_ylabel("Count")
for bar, val in zip(bars, counts):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+300,
                 f"{val:,}\n({val/sum(counts)*100:.1f}%)", ha="center", fontsize=10)
axes[1].hist(shot_logs["SHOT_DIST"].dropna(), bins=50, color="#3498db", edgecolor="k", lw=0.4, alpha=0.85)
axes[1].set_title("Shot Distance Distribution", fontweight="bold")
axes[1].set_xlabel("Distance (feet)"); axes[1].set_ylabel("Count")
for label, colour, name in [("made","#2ecc71","Made"),("missed","#e74c3c","Missed")]:
    mask = shot_logs["SHOT_RESULT"] == label
    axes[2].hist(shot_logs.loc[mask,"CLOSE_DEF_DIST"].dropna(), bins=40,
                 alpha=0.6, color=colour, label=name, edgecolor="k", lw=0.3)
axes[2].set_title("Defender Distance by Outcome", fontweight="bold")
axes[2].set_xlabel("Closest Defender (ft)"); axes[2].legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/eda_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 2: FG% by period
period_fg = shot_logs.groupby("PERIOD").apply(lambda x: (x["SHOT_RESULT"]=="made").mean()).reset_index()
period_fg.columns = ["PERIOD","FG_PCT"]
fig, ax = plt.subplots(figsize=(8,5))
bars = ax.bar(period_fg["PERIOD"].astype(str), period_fg["FG_PCT"]*100,
              color="#9b59b6", edgecolor="k", lw=0.8)
ax.set_title("Field Goal % by Period", fontweight="bold"); ax.set_ylabel("FG%"); ax.set_ylim(0,60)
for bar, val in zip(bars, period_fg["FG_PCT"]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{val*100:.1f}%", ha="center")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/eda_fg_by_period.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 3: Correlation heatmap
corr_cols = ["SHOT_DIST","CLOSE_DEF_DIST","SHOT_CLOCK","TOUCH_TIME","DRIBBLES","PTS_TYPE",
             "PERIOD","FINAL_MARGIN","FGM"]
corr_df = shot_logs[corr_cols + ["SHOT_RESULT"]].copy()
corr_df["SHOT_RESULT"] = (corr_df["SHOT_RESULT"]=="made").astype(int)
corr_df["SHOT_CLOCK"].fillna(corr_df["SHOT_CLOCK"].median(), inplace=True)
fig, ax = plt.subplots(figsize=(12,10))
mask = np.triu(np.ones_like(corr_df.corr(), dtype=bool))
sns.heatmap(corr_df.corr(), mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            ax=ax, linewidths=0.5, square=True, cbar_kws={"shrink":0.8})
ax.set_title("Feature Correlation Heatmap", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/eda_correlation.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 4: Rolling FG% trend for one player
sample_pid = df_model["PLAYER_ID"].value_counts().index[0]
sp = df_model[df_model["PLAYER_ID"]==sample_pid].copy()
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(range(len(sp)), sp["ROLL_FG_5"].values, color="#e67e22", lw=1.5, label="Rolling FG% (5-shot)")
ax.scatter(range(len(sp)), sp["TARGET"].values * sp["ROLL_FG_5"].max(),
           c=["#2ecc71" if t==1 else "#e74c3c" for t in sp["TARGET"]], s=15, alpha=0.5, zorder=3)
ax.set_title(f"Rolling FG% Trend — Player {sample_pid}", fontweight="bold")
ax.set_xlabel("Shot Index"); ax.set_ylabel("Scaled Rolling FG%"); ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/eda_rolling_fg_trend.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved 4 plots → {OUTPUT_DIR}/plots/")

# ── STEP 12: SAVE ────────────────────────────────────────────
print("\n[STEP 12] Saving outputs …")
np.save(f"{OUTPUT_DIR}/X_sequences.npy", X_sequences)
np.save(f"{OUTPUT_DIR}/y_labels.npy",    y_labels)
np.save(f"{OUTPUT_DIR}/X_flat.npy",      X_flat)
df_model.to_csv(f"{OUTPUT_DIR}/shot_logs_merged.csv", index=False)
with open(f"{OUTPUT_DIR}/scaler.pkl","wb") as f: pickle.dump(scaler, f)
with open(f"{OUTPUT_DIR}/feature_names.pkl","wb") as f: pickle.dump(SEQUENCE_FEATURES, f)
metadata = {
    "seq_len":SEQ_LEN, "n_features":len(SEQUENCE_FEATURES),
    "n_samples":int(X_sequences.shape[0]),
    "n_made":int(y_labels.sum()), "n_missed":int(len(y_labels)-y_labels.sum()),
    "class_balance":float(y_labels.mean()), "feature_names":SEQUENCE_FEATURES,
    "sequence_shape":X_sequences.shape,
}
with open(f"{OUTPUT_DIR}/preprocessing_metadata.pkl","wb") as f: pickle.dump(metadata, f)

print(f"  X_sequences.npy      → {X_sequences.shape}")
print(f"  y_labels.npy         → {y_labels.shape}")
print(f"  X_flat.npy           → {X_flat.shape}")
print(f"  shot_logs_merged.csv → {df_model.shape}")
print(f"  scaler.pkl, feature_names.pkl, preprocessing_metadata.pkl  ✓")

print("\n" + "="*65)
print("  PART 1 COMPLETE  ✓   Run part2_training.py next.")
print("="*65)
print(f"  Sequences: {X_sequences.shape[0]:,}  |  SeqLen: {SEQ_LEN}  "
      f"|  Features: {len(SEQUENCE_FEATURES)}")
print(f"  Made: {y_labels.sum():,} ({y_labels.mean()*100:.1f}%)  "
      f"Missed: {(1-y_labels).sum():,} ({(1-y_labels.mean())*100:.1f}%)\n")
