"""
Automatic backend selection for GP fitting and scoring.

Chooses optimal backends based on problem size and available hardware.
"""
from __future__ import annotations

from typing import Literal

BackendType = Literal["python", "native_lapack", "native_reference", "gpu", "daemon"]


def select_fit_backend(
    m: int,
    manual_override: str | None = None,
    optimize_hyperparameters: bool = False,
) -> str:
    """
    Select optimal fit backend based on problem size.

    Args:
        m: Number of rated points (training set size)
        manual_override: User-specified backend (bypasses auto-selection)
        optimize_hyperparameters: Whether HP optimization is requested

    Returns:
        Backend name: "python", "native_lapack", or "native_reference"

    Decision tree:
        - manual_override: Use specified backend (no auto-selection)
        - optimize_hyperparameters: Must use "python" (only backend with HP support)
        - m < 5000 + native available: "native_lapack" (instant, zero overhead)
        - m < 10000: "python" (scipy, good enough)
        - m >= 10000: "native_reference" (ScaLAPACK MPI for large problems)
    """
    # Manual override takes precedence
    if manual_override and manual_override != "auto":
        return manual_override

    # Hyperparameter optimization requires python backend
    if optimize_hyperparameters:
        return "python"

    # Check availability
    try:
        from .native_lapack import is_native_available
        has_native = is_native_available()
    except ImportError:
        has_native = False

    # Auto-selection based on problem size
    if m < 5000 and has_native:
        return "native_lapack"  # Best for small-medium: instant fit
    elif m < 10000:
        return "python"  # Good enough for medium problems
    else:
        return "native_reference"  # ScaLAPACK for large problems


def select_score_backend(
    n: int,
    m: int,
    manual_override: str | None = None,
) -> str:
    """
    Select optimal scoring backend based on problem size and hardware.

    Args:
        n: Number of candidates to score
        m: Number of rated points (affects variance computation cost)
        manual_override: User-specified backend (bypasses auto-selection)

    Returns:
        Backend name: "python", "native_lapack", or "gpu"

    Decision tree:
        - manual_override: Use specified backend (no auto-selection)
        - GPU available + m >= 500: "gpu" (3-4.6× faster)
        - native_lapack available: "native_lapack" (1.1-1.2× faster, zero overhead)
        - fallback: "python" (always works)

    Performance data (from benchmarks):
        m < 500: native_lapack best (GPU cold-start overhead)
        m = 500-5k: GPU best (3-4.6× faster)
        m > 5k: GPU best (2.3-2.7× faster)
        No GPU: native_lapack (1.1-1.2× faster than Python)
    """
    # Manual override takes precedence
    if manual_override and manual_override != "auto":
        return manual_override

    # Check availability
    try:
        from .gpu_scoring import is_gpu_available
        has_gpu = is_gpu_available()
    except ImportError:
        has_gpu = False

    try:
        from .native_lapack import is_native_available
        has_native = is_native_available()
    except ImportError:
        has_native = False

    # Auto-selection based on problem size and hardware
    if has_gpu and m >= 500:
        # GPU dominates for m >= 500 (overcomes cold-start overhead)
        return "gpu"
    elif has_native:
        # native_lapack is best CPU option (zero overhead, BLAS optimized)
        return "native_lapack"
    else:
        # Fallback to scipy (always available)
        return "python"


def get_backend_info() -> dict[str, bool]:
    """
    Get availability status of all backends.

    Returns:
        Dictionary with backend availability:
            - native_lapack: PyBind11 LAPACK available
            - gpu: CuPy/CUDA available
            - scalapack: MPI/ScaLAPACK available (always True)
    """
    try:
        from .native_lapack import is_native_available
        has_native = is_native_available()
    except ImportError:
        has_native = False

    try:
        from .gpu_scoring import is_gpu_available
        has_gpu = is_gpu_available()
    except ImportError:
        has_gpu = False

    return {
        "native_lapack": has_native,
        "gpu": has_gpu,
        "scalapack": True,  # Always available (native_reference)
        "python": True,  # Always available
    }


def print_backend_status(verbose: bool = True) -> None:
    """
    Print backend availability and recommendations.

    Args:
        verbose: If True, print detailed recommendations
    """
    info = get_backend_info()

    print("Backend Availability:")
    print(f"  Python (scipy):     ✓ Always available")
    print(f"  Native LAPACK:      {'✓ Available' if info['native_lapack'] else '✗ Not available (build with: make native-build)'}")
    print(f"  GPU (CuPy):         {'✓ Available' if info['gpu'] else '✗ Not available (install cupy-cuda11x/12x)'}")
    print(f"  ScaLAPACK (MPI):    ✓ Always available")

    if verbose:
        print()
        print("Recommendations:")
        print("  Fit backend:")
        print("    m < 5k:   native_lapack (instant) or python")
        print("    m < 10k:  python")
        print("    m >= 10k: native_reference (ScaLAPACK MPI)")
        print()
        print("  Score backend:")
        if info['gpu']:
            print("    m < 500:  native_lapack (GPU cold-start overhead)")
            print("    m >= 500: gpu (3-4× faster)")
        elif info['native_lapack']:
            print("    Any m:    native_lapack (1.1-1.2× faster)")
        else:
            print("    Any m:    python (baseline)")
        print()
        print("  Use fit_backend='auto' or score_backend='auto' for automatic selection")
