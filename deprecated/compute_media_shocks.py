# compute_media_shocks.py
#
# Converts posterior M_draws into log-difference media shocks.
#
# Input:
#   data/media_better_draws.npz
#
# Output:
#   data/media_shock_draws.npz

import numpy as np
import os

INPUT_PATH = "data/media_better_draws.npz"
OUTPUT_PATH = "data/media_shock_draws.npz"

data = np.load(INPUT_PATH, allow_pickle=True)

M_draws = data["M_draws"]          # shape (S, T)
year_month = data["year_month"]

# Convert to log scale
logM_draws = np.log(M_draws)

# Compute month-to-month differences
# shock[:, t] = logM_t - logM_{t-1}
shock_draws = np.zeros_like(logM_draws)
shock_draws[:, 1:] = logM_draws[:, 1:] - logM_draws[:, :-1]
shock_draws[:, 0] = 0.0   # no prior month

np.savez_compressed(
    OUTPUT_PATH,
    year_month=year_month,
    shock_draws=shock_draws,
    logM_draws=logM_draws,
)

print(f"Wrote: {OUTPUT_PATH}")
print("Shape:", shock_draws.shape)