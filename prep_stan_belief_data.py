# prep_stan_belief_data.py
#
# Prepares monthly belief model data (pre-decision window only)
#
# Input:
#   data/historical_belief_table.csv
#
# Output:
#   data/stan_belief_data.json

import numpy as np
import pandas as pd
import json

INPUT = "data/historical_belief_table.csv"
OUTPUT = "data/stan_belief_data.json"

df = pd.read_csv(INPUT)
df = df.sort_values("year_month").reset_index(drop=True)

T = len(df)

# Media shock: z-score for the Stan model
shock_raw = df["media_shock_mean"].values
shock_mean = float(np.nanmean(shock_raw))
shock_sd = float(np.nanstd(shock_raw, ddof=0))
shock = ((shock_raw - shock_mean) / shock_sd if shock_sd > 0 else shock_raw * 0.0)

# Abnormal returns: z-score
abret_raw = df["abret_sum"].values
abret_mean = float(np.nanmean(abret_raw))
abret_sd = float(np.nanstd(abret_raw, ddof=0))
abret = ((abret_raw - abret_mean) / abret_sd if abret_sd > 0 else abret_raw * 0.0)

# Remuneration votes (only where not NaN)
rem_mask = df["vote_against_rem_pct"].notna().values
rem_idx = np.where(rem_mask)[0] + 1  # 1-based
rem_obs = df.loc[rem_mask, "vote_against_rem_pct"].values

# Chair votes
chair_mask = df["vote_against_chair_pct"].notna().values
chair_idx = np.where(chair_mask)[0] + 1
chair_obs = df.loc[chair_mask, "vote_against_chair_pct"].values

data = {
    "T": int(T),
    "shock": shock.tolist(),
    "abret": abret.tolist(),
    "shock_mean": shock_mean,
    "shock_sd": shock_sd,
    "abret_mean": abret_mean,
    "abret_sd": abret_sd,

    "N_rem": int(len(rem_idx)),
    "rem_idx": rem_idx.astype(int).tolist(),
    "rem_obs": rem_obs.tolist(),

    "N_chair": int(len(chair_idx)),
    "chair_idx": chair_idx.astype(int).tolist(),
    "chair_obs": chair_obs.tolist(),
}

with open(OUTPUT, "w") as f:
    json.dump(data, f, indent=2)

print("Wrote:", OUTPUT)
print("T =", T)
print("N_rem =", len(rem_idx))
print("N_chair =", len(chair_idx))