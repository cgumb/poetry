"""
Configuration for Poetry GP backends and hyperparameters.

Allows users to override automatic backend selection and set preferences.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class GPConfig:
    """
    Configuration for GP fitting and scoring.

    Use 'auto' for automatic backend selection based on problem size,
    or specify a backend explicitly to override automatic selection.
    """

    # Backend selection
    fit_backend: str = "auto"  # "auto", "python", "native_lapack", "native_reference"
    score_backend: str = "auto"  # "auto", "python", "native_lapack", "gpu", "daemon", "none"

    # GP hyperparameters
    length_scale: float = 1.0
    variance: float = 1.0
    noise: float = 1e-3

    # Acquisition strategies
    exploitation_strategy: str = "max_mean"  # "max_mean", "ucb", "lcb", "thompson"
    exploration_strategy: str = "max_variance"  # "max_variance", "spatial_variance", "expected_improvement"
    ucb_beta: float = 2.0  # Confidence parameter for UCB/LCB

    # Performance tuning
    block_size: int = 2048
    optimize_hyperparameters: bool = False
    optimizer_maxiter: int = 50

    # MPI/ScaLAPACK options (for native_reference fit backend)
    scalapack_nprocs: int = 4
    scalapack_launcher: str = "srun"
    scalapack_block_size: int = 128

    # Daemon options (for daemon score backend)
    daemon_nprocs: int = 4
    daemon_launcher: str = "mpirun"

    def to_dict(self) -> dict:
        """Convert config to dictionary for **kwargs passing."""
        return {
            "fit_backend": self.fit_backend,
            "score_backend": self.score_backend,
            "length_scale": self.length_scale,
            "variance": self.variance,
            "noise": self.noise,
            "exploitation_strategy": self.exploitation_strategy,
            "exploration_strategy": self.exploration_strategy,
            "ucb_beta": self.ucb_beta,
            "block_size": self.block_size,
            "optimize_hyperparameters": self.optimize_hyperparameters,
            "optimizer_maxiter": self.optimizer_maxiter,
            "scalapack_nprocs": self.scalapack_nprocs,
            "scalapack_launcher": self.scalapack_launcher,
            "scalapack_block_size": self.scalapack_block_size,
            "daemon_nprocs": self.daemon_nprocs,
            "daemon_launcher": self.daemon_launcher,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GPConfig:
        """Create config from dictionary."""
        # Filter to only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)

    def copy(self, **overrides) -> GPConfig:
        """Create a copy with optional field overrides."""
        d = self.to_dict()
        d.update(overrides)
        return self.from_dict(d)


# Default configuration
DEFAULT_CONFIG = GPConfig()


# Performance-tuned presets
FAST_CONFIG = GPConfig(
    fit_backend="native_lapack",
    score_backend="auto",  # GPU if available, else native_lapack
    optimize_hyperparameters=False,
)

ACCURATE_CONFIG = GPConfig(
    fit_backend="auto",
    score_backend="auto",
    optimize_hyperparameters=True,
    optimizer_maxiter=100,
    noise=1e-4,  # Lower noise for more accuracy
)

LARGE_SCALE_CONFIG = GPConfig(
    fit_backend="native_reference",  # ScaLAPACK MPI for m > 10k
    score_backend="gpu",  # GPU for n > 10k
    scalapack_nprocs=16,
    block_size=4096,
)


def print_config(config: GPConfig) -> None:
    """Pretty-print configuration."""
    print("GP Configuration:")
    print(f"  Fit backend:         {config.fit_backend}")
    print(f"  Score backend:       {config.score_backend}")
    print(f"  Length scale:        {config.length_scale}")
    print(f"  Variance:            {config.variance}")
    print(f"  Noise:               {config.noise}")
    print(f"  Exploitation:        {config.exploitation_strategy}")
    print(f"  Exploration:         {config.exploration_strategy}")
    print(f"  Block size:          {config.block_size}")
    print(f"  Optimize HP:         {config.optimize_hyperparameters}")
    if config.optimize_hyperparameters:
        print(f"    Max iterations:    {config.optimizer_maxiter}")
