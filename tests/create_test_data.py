"""
Generate synthetic checkpoint .npz files for testing.

Creates realistic-looking posterior draws that allow the full engine
to be tested without running Stan.
"""

import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np


def create_synthetic_checkpoint(
    checkpoint_id: str,
    checkpoint_date: str,
    output_dir: str | Path,
    N: int = 500,
    seed: int = 42,
    belief_level: float = 0.0,
) -> Path:
    """
    Create a synthetic checkpoint .npz file.

    Args:
        checkpoint_id: e.g. "C0"
        checkpoint_date: e.g. "2023-10-01"
        output_dir: Directory to write the file.
        N: Number of draws.
        seed: Random seed.
        belief_level: Mean belief level (higher = more distrust).

    Returns:
        Path to created file.
    """
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # B_mkt: market-visible belief state (AR1 draws around belief_level)
    B_mkt = rng.normal(belief_level, 0.5, size=N)

    # B_mgmt: management-known belief state (slightly shifted, more informed)
    B_mgmt = B_mkt + rng.normal(0.2, 0.1, size=N)

    # Vote model parameters
    alpha_vote = rng.normal(-1.5, 0.3, size=N)  # Base log-odds against
    gamma_A = rng.normal(0.5, 0.2, size=N)      # ASA mobilisation effect
    gamma_D = rng.normal(-0.3, 0.1, size=N)     # Governance package dampening
    sigma_vote = np.abs(rng.normal(0.5, 0.1, size=N))  # Vote noise

    # Review model parameters
    review_param_1 = rng.normal(-0.5, 0.3, size=N)  # Base review log-odds
    review_param_2 = rng.normal(0.3, 0.1, size=N)   # Context adjustment

    # Draw IDs
    draw_id = np.arange(N, dtype=np.int64)

    # Metadata
    metadata = {
        "schema_version": "1.0",
        "checkpoint_id": checkpoint_id,
        "checkpoint_date": checkpoint_date,
        "n_draws": N,
        "stan_model_versions": {
            "belief_model": "v1.0.0-synthetic",
            "vote_model": "v1.0.0-synthetic",
            "review_model": "v1.0.0-synthetic",
        },
        "priors_used": {
            "gamma_A_prior_source": "synthetic",
            "review_prior_source": "synthetic",
        },
        "ar1_parameters_summary": {
            "rho_mean": 0.82,
            "sigma_B_mean": 0.14,
        },
        "random_seed": seed,
        "generation_timestamp_utc": datetime.utcnow().isoformat() + "Z",
    }

    filepath = output_dir / f"belief_{checkpoint_id}_{checkpoint_date}.npz"
    np.savez(
        filepath,
        B_mkt=B_mkt,
        B_mgmt=B_mgmt,
        alpha_vote=alpha_vote,
        gamma_A=gamma_A,
        gamma_D=gamma_D,
        sigma_vote=sigma_vote,
        review_param_1=review_param_1,
        review_param_2=review_param_2,
        draw_id=draw_id,
        metadata_json=json.dumps(metadata),
    )

    return filepath


def create_all_test_checkpoints(output_dir: str | Path, N: int = 500) -> list[Path]:
    """Create all five test checkpoint files."""
    configs = [
        ("Cpre", "2023-08-31", -0.1, 99),  # Pre-crisis: ACCC action, mild concern
        ("C0", "2023-10-01", 0.0, 100),
        ("C1", "2023-10-10", 0.3, 101),
        ("C2", "2023-10-18", 0.6, 102),
        ("C3", "2023-11-03", 1.5, 103),  # High distrust at AGM
    ]

    paths = []
    for cid, date, belief, seed in configs:
        p = create_synthetic_checkpoint(
            checkpoint_id=cid,
            checkpoint_date=date,
            output_dir=output_dir,
            N=N,
            seed=seed,
            belief_level=belief,
        )
        paths.append(p)
        print(f"Created {p}")

    return paths


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    checkpoint_dir = project_root / "data" / "checkpoints"
    create_all_test_checkpoints(checkpoint_dir)
    print("All test checkpoints created.")
