from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from poetry_gp.blr import fit_bayesian_linear_regression, predict_bayesian_linear_regression

POET_CANDIDATES = ["poet", "author", "poet_name"]


def pick_column(columns: list[str], candidates: list[str], fallback: str) -> str:
    for c in candidates:
        if c in columns:
            return c
    return fallback


def parse_poet_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def synthetic_targets(x: np.ndarray, *, seed: int, noise_std: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    beta_true = rng.normal(size=x.shape[1])
    beta_true /= np.linalg.norm(beta_true) + 1e-12
    y = x @ beta_true + rng.normal(scale=noise_std, size=x.shape[0])
    return y, beta_true


def poet_proxy_targets(df: pd.DataFrame, poet_col: str, positive_poets: list[str], negative_poets: list[str]) -> tuple[np.ndarray, np.ndarray]:
    poet_series = df[poet_col].astype(str)
    pos = poet_series.isin(positive_poets).to_numpy()
    neg = poet_series.isin(negative_poets).to_numpy()
    mask = pos | neg
    y = np.where(pos[mask], 1.0, -1.0)
    return mask, y.astype(np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poems", type=Path, default=Path("data/poems.parquet"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings.npy"))
    parser.add_argument("--mode", choices=["synthetic_linear", "poet_proxy"], default="synthetic_linear")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-std", type=float, default=0.25)
    parser.add_argument("--prior-precision", type=float, default=1.0)
    parser.add_argument("--noise-variance", type=float, default=1.0)
    parser.add_argument("--positive-poets", default="")
    parser.add_argument("--negative-poets", default="")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    poems = pd.read_parquet(args.poems)
    x = np.asarray(np.load(args.embeddings, mmap_mode="r"), dtype=np.float64)
    if len(poems) != x.shape[0]:
        raise ValueError("poem and embedding row counts do not match")

    rng = np.random.default_rng(args.seed)
    poet_col = pick_column(list(poems.columns), POET_CANDIDATES, poems.columns[0])

    if args.mode == "synthetic_linear":
        y, beta_true = synthetic_targets(x, seed=args.seed, noise_std=args.noise_std)
        candidate_indices = np.arange(len(y), dtype=np.int64)
        metadata = {"beta_true_norm": float(np.linalg.norm(beta_true))}
    else:
        positive_poets = parse_poet_list(args.positive_poets)
        negative_poets = parse_poet_list(args.negative_poets)
        if not positive_poets or not negative_poets:
            raise ValueError("poet_proxy mode requires both --positive-poets and --negative-poets")
        mask, y = poet_proxy_targets(poems, poet_col, positive_poets, negative_poets)
        candidate_indices = np.flatnonzero(mask).astype(np.int64)
        x = x[candidate_indices]
        metadata = {
            "positive_poets": positive_poets,
            "negative_poets": negative_poets,
            "n_proxy_poems": int(len(candidate_indices)),
        }
        if len(y) < 4:
            raise ValueError("poet_proxy mode produced too few labeled poems")

    order = rng.permutation(len(y))
    n_train = max(1, min(len(y) - 1, int(round(args.train_fraction * len(y)))))
    train_idx = order[:n_train]
    test_idx = order[n_train:]

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    state = fit_bayesian_linear_regression(
        x_train,
        y_train,
        prior_precision=args.prior_precision,
        noise_variance=args.noise_variance,
        fit_intercept=True,
    )
    train_mean, train_var = predict_bayesian_linear_regression(state, x_train)
    test_mean, test_var = predict_bayesian_linear_regression(state, x_test)

    result = {
        "mode": args.mode,
        "n_total": int(len(y)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "dim": int(x.shape[1]),
        "prior_precision": float(args.prior_precision),
        "noise_variance": float(args.noise_variance),
        "train_rmse": float(np.sqrt(np.mean((train_mean - y_train) ** 2))),
        "test_rmse": float(np.sqrt(np.mean((test_mean - y_test) ** 2))),
        "train_predictive_variance_mean": float(np.mean(train_var)),
        "test_predictive_variance_mean": float(np.mean(test_var)),
        "metadata": metadata,
    }

    text = json.dumps(result, indent=2)
    if args.output is None:
        print(text)
    else:
        args.output.write_text(text)
        print(f"wrote BLR baseline result to {args.output}")


if __name__ == "__main__":
    main()
