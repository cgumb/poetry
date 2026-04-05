from __future__ import annotations

import json
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..gp_exact import GPState


@dataclass
class ScaLAPACKFitResult:
    alpha: np.ndarray
    chol_lower: np.ndarray
    logdet: float
    factor_seconds: float
    solve_seconds: float
    gather_seconds: float
    total_seconds: float
    block_size: int
    grid_rows: int | None
    grid_cols: int | None
    info_potrf: int
    info_potrs: int
    implemented: bool
    backend: str
    requested_backend: str
    compiled_with_scalapack: bool
    message: str
    workdir: Path


@dataclass
class ScaLAPACKPreparedRun:
    workdir: Path
    command: list[str]
    input_meta_path: Path
    matrix_bin_path: Path
    rhs_bin_path: Path
    output_meta_path: Path
    alpha_bin_path: Path
    chol_bin_path: Path


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _check_problem_size_and_warn(n: int, nprocs: int) -> None:
    """
    Warn users when ScaLAPACK overhead dominates computation.

    Current overhead (~2-3s after BLAS optimization):
    - Subprocess launch + MPI init: ~0.3-0.5s
    - Kernel assembly (BLAS-optimized): ~0.1-0.5s
    - Scatter/gather: ~1.0-1.5s
    - File I/O: ~0.5s

    Python's O(n³) Cholesky becomes competitive around n=5000-10000.
    After Milestone 1B (distributed assembly), crossover drops to n=1000-2000.
    """
    # Warn for very small problems that will definitely be slower
    if n < 1000:
        warnings.warn(
            f"ScaLAPACK backend for n={n} is likely slower than Python due to overhead "
            f"(subprocess launch, MPI init, scatter/gather ~2-3s). "
            f"Consider using fit_backend='python' for n < 1000. "
            f"See docs/NATIVE_HPC_ROADMAP.md for details.",
            UserWarning,
            stacklevel=3,
        )
    # Inform for medium problems where performance may be mixed
    elif n < 5000 and nprocs > 1:
        warnings.warn(
            f"ScaLAPACK backend for n={n} with {nprocs} processes may not be faster than Python. "
            f"Current overhead is ~2-3s. Crossover point is around n=5000-10000. "
            f"For best performance on small problems, use fit_backend='python'. "
            f"See docs/BENCHMARKING_GUIDE.md for performance analysis.",
            UserWarning,
            stacklevel=3,
        )


def _build_launcher_command(launcher: str, nprocs: int, executable: str) -> list[str]:
    if launcher == "srun":
        return [launcher, "-n", str(nprocs), executable]
    if launcher == "mpirun":
        return [launcher, "--bind-to", "none", "--map-by", "slot", "-np", str(nprocs), executable]
    if launcher in {"local", "none"}:
        if nprocs != 1:
            raise ValueError("launcher='local' requires nprocs=1")
        return [executable]
    raise ValueError("launcher must be 'srun', 'mpirun', or 'local'")


def prepare_scalapack_fit_workdir(
    K_rr: np.ndarray,
    y: np.ndarray,
    *,
    launcher: str = "srun",
    nprocs: int = 4,
    executable: str = "native/build/scalapack_gp_fit",
    block_size: int = 128,
    grid_rows: int | None = None,
    grid_cols: int | None = None,
    native_backend: str = "auto",
    return_alpha: bool = True,
    return_chol: bool = True,
    workdir: Path | None = None,
) -> ScaLAPACKPreparedRun:
    K_rr = np.asarray(K_rr, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if K_rr.ndim != 2 or K_rr.shape[0] != K_rr.shape[1]:
        raise ValueError("K_rr must be a square 2D array")
    if y.ndim != 1 or y.shape[0] != K_rr.shape[0]:
        raise ValueError("y must be a 1D array with the same length as K_rr")

    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="scalapack_fit_"))
    workdir.mkdir(parents=True, exist_ok=True)

    input_meta_path = workdir / "input_meta.json"
    matrix_bin_path = workdir / "K_rr.bin"
    rhs_bin_path = workdir / "y.bin"
    output_meta_path = workdir / "output_meta.json"
    alpha_bin_path = workdir / "alpha.bin"
    chol_bin_path = workdir / "chol_lower.bin"

    meta = {
        "input_kind": "matrix",
        "n": int(K_rr.shape[0]),
        "rhs_cols": 1,
        "dtype": "float64",
        "matrix_layout": "row_major",
        "matrix_triangle": "lower",
        "block_size": int(block_size),
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "noise_included": True,
    }
    _write_json(input_meta_path, meta)
    K_rr.astype(np.float64, copy=False).tofile(matrix_bin_path)
    y.astype(np.float64, copy=False).tofile(rhs_bin_path)

    command = _build_launcher_command(launcher, nprocs, executable)
    command += [
        "--input-meta",
        str(input_meta_path),
        "--matrix-bin",
        str(matrix_bin_path),
        "--rhs-bin",
        str(rhs_bin_path),
        "--output-meta",
        str(output_meta_path),
        "--alpha-bin",
        str(alpha_bin_path),
        "--chol-bin",
        str(chol_bin_path),
        "--backend",
        native_backend,
        "--block-size",
        str(block_size),
        "--return-alpha",
        "1" if return_alpha else "0",
        "--return-chol",
        "1" if return_chol else "0",
    ]

    return ScaLAPACKPreparedRun(
        workdir=workdir,
        command=command,
        input_meta_path=input_meta_path,
        matrix_bin_path=matrix_bin_path,
        rhs_bin_path=rhs_bin_path,
        output_meta_path=output_meta_path,
        alpha_bin_path=alpha_bin_path,
        chol_bin_path=chol_bin_path,
    )


def prepare_scalapack_feature_fit_workdir(
    x_rated: np.ndarray,
    y: np.ndarray,
    *,
    length_scale: float,
    variance: float,
    noise: float,
    launcher: str = "srun",
    nprocs: int = 4,
    executable: str = "native/build/scalapack_gp_fit",
    block_size: int = 128,
    grid_rows: int | None = None,
    grid_cols: int | None = None,
    native_backend: str = "auto",
    return_alpha: bool = True,
    return_chol: bool = True,
    workdir: Path | None = None,
) -> ScaLAPACKPreparedRun:
    x_rated = np.asarray(x_rated, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x_rated.ndim != 2:
        raise ValueError("x_rated must be 2D")
    if y.ndim != 1 or y.shape[0] != x_rated.shape[0]:
        raise ValueError("y must be a 1D array with the same length as x_rated")

    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="scalapack_fit_"))
    workdir.mkdir(parents=True, exist_ok=True)

    input_meta_path = workdir / "input_meta.json"
    matrix_bin_path = workdir / "x_rated.bin"
    rhs_bin_path = workdir / "y.bin"
    output_meta_path = workdir / "output_meta.json"
    alpha_bin_path = workdir / "alpha.bin"
    chol_bin_path = workdir / "chol_lower.bin"

    meta = {
        "input_kind": "features",
        "n": int(x_rated.shape[0]),
        "d": int(x_rated.shape[1]),
        "rhs_cols": 1,
        "dtype": "float64",
        "matrix_layout": "row_major",
        "matrix_triangle": "lower",
        "block_size": int(block_size),
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "noise_included": True,
        "length_scale": float(length_scale),
        "variance": float(variance),
        "noise": float(noise),
    }
    _write_json(input_meta_path, meta)
    x_rated.astype(np.float64, copy=False).tofile(matrix_bin_path)
    y.astype(np.float64, copy=False).tofile(rhs_bin_path)

    command = _build_launcher_command(launcher, nprocs, executable)
    command += [
        "--input-meta",
        str(input_meta_path),
        "--matrix-bin",
        str(matrix_bin_path),
        "--rhs-bin",
        str(rhs_bin_path),
        "--output-meta",
        str(output_meta_path),
        "--alpha-bin",
        str(alpha_bin_path),
        "--chol-bin",
        str(chol_bin_path),
        "--backend",
        native_backend,
        "--block-size",
        str(block_size),
        "--return-alpha",
        "1" if return_alpha else "0",
        "--return-chol",
        "1" if return_chol else "0",
    ]

    return ScaLAPACKPreparedRun(
        workdir=workdir,
        command=command,
        input_meta_path=input_meta_path,
        matrix_bin_path=matrix_bin_path,
        rhs_bin_path=rhs_bin_path,
        output_meta_path=output_meta_path,
        alpha_bin_path=alpha_bin_path,
        chol_bin_path=chol_bin_path,
    )


def _run_prepared_fit(
    prepared: ScaLAPACKPreparedRun,
    *,
    native_backend: str,
    block_size: int,
    verbose: bool,
) -> ScaLAPACKFitResult:
    if verbose:
        print(f"[native-fit] workdir: {prepared.workdir}", flush=True)
        print(f"[native-fit] command: {' '.join(prepared.command)}", flush=True)
    completed = subprocess.run(prepared.command, check=False, capture_output=True, text=True)
    if verbose:
        print(f"[native-fit] launcher return code: {completed.returncode}", flush=True)
        if completed.stdout:
            print(completed.stdout, end="", flush=True)
        if completed.stderr:
            print(completed.stderr, end="", flush=True)

    if not prepared.output_meta_path.exists():
        pieces = [
            "Native fit command failed before writing output metadata.",
            f"command={' '.join(prepared.command)}",
            f"returncode={completed.returncode}",
            f"workdir={prepared.workdir}",
        ]
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        if stdout:
            pieces.append(f"stdout={stdout}")
        if stderr:
            pieces.append(f"stderr={stderr}")
        raise RuntimeError(" | ".join(pieces))

    # Read and parse metadata JSON with better error handling
    if not prepared.output_meta_path.exists():
        raise RuntimeError(
            f"ScaLAPACK executable succeeded but did not create metadata file: {prepared.output_meta_path}\n"
            f"stdout: {completed.stdout}\n"
            f"stderr: {completed.stderr}"
        )

    meta_text = prepared.output_meta_path.read_text()
    if not meta_text.strip():
        raise RuntimeError(
            f"ScaLAPACK executable created empty metadata file: {prepared.output_meta_path}\n"
            f"stdout: {completed.stdout}\n"
            f"stderr: {completed.stderr}"
        )

    try:
        meta = json.loads(meta_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse ScaLAPACK metadata JSON: {e}\n"
            f"File: {prepared.output_meta_path}\n"
            f"Content: {meta_text[:500]}\n"
            f"stdout: {completed.stdout}\n"
            f"stderr: {completed.stderr}"
        ) from e

    n = int(meta["n"])
    has_alpha = bool(meta.get("has_alpha", True))  # Default True for backward compatibility
    has_chol = bool(meta.get("has_chol", True))

    # Only read outputs that were actually gathered
    if has_alpha:
        alpha = np.fromfile(prepared.alpha_bin_path, dtype=np.float64, count=n)
    else:
        alpha = np.zeros(n, dtype=np.float64)

    if has_chol:
        chol = np.fromfile(prepared.chol_bin_path, dtype=np.float64, count=n * n).reshape(n, n)
    else:
        chol = np.zeros((n, n), dtype=np.float64)

    result = ScaLAPACKFitResult(
        alpha=alpha,
        chol_lower=chol,
        logdet=float(meta.get("logdet", 0.0)),
        factor_seconds=float(meta.get("factor_seconds", 0.0)),
        solve_seconds=float(meta.get("solve_seconds", 0.0)),
        gather_seconds=float(meta.get("gather_seconds", 0.0)),
        total_seconds=float(meta.get("total_seconds", 0.0)),
        block_size=int(block_size),
        grid_rows=meta.get("grid_rows"),
        grid_cols=meta.get("grid_cols"),
        info_potrf=int(meta.get("info_potrf", -1)),
        info_potrs=int(meta.get("info_potrs", -1)),
        implemented=bool(meta.get("implemented", False)),
        backend=str(meta.get("backend", "unknown")),
        requested_backend=str(meta.get("requested_backend", native_backend)),
        compiled_with_scalapack=bool(meta.get("compiled_with_scalapack", False)),
        message=str(meta.get("message", "")),
        workdir=prepared.workdir,
    )
    if verbose:
        print(
            "[native-fit] output meta: "
            f"backend={result.backend} requested={result.requested_backend} "
            f"implemented={result.implemented} info_potrf={result.info_potrf} info_potrs={result.info_potrs}",
            flush=True,
        )

    # Check computation success first (before checking exit code)
    # This allows us to tolerate cleanup crashes that happen after computation completes
    if result.info_potrf != 0 or result.info_potrs not in {0, -1}:
        raise RuntimeError(result.message)
    if not result.implemented:
        raise NotImplementedError(result.message)

    # Only raise on non-zero exit if computation actually failed
    # Exit code 139 (segfault during cleanup) is OK if info codes are good
    if completed.returncode != 0 and result.implemented:
        # If computation succeeded (info codes good), just warn about exit code
        if result.info_potrf == 0 and result.info_potrs in {0, -1}:
            if verbose or completed.returncode != 139:
                # 139 = 128 + 11 (SIGSEGV) is a known cleanup issue
                warnings.warn(
                    f"Native backend exited with code {completed.returncode} but computation succeeded "
                    f"(info_potrf={result.info_potrf}, info_potrs={result.info_potrs}). "
                    f"This is likely a harmless cleanup crash in MPI/BLACS destructors.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            # Computation actually failed
            if completed.stderr:
                print("[native-fit] stderr from failed run:", flush=True)
                print(completed.stderr, flush=True)
            raise RuntimeError(result.message)

    return result


def fit_exact_gp_scalapack(
    K_rr: np.ndarray,
    y: np.ndarray,
    *,
    launcher: str = "srun",
    nprocs: int = 4,
    executable: str = "native/build/scalapack_gp_fit",
    block_size: int = 128,
    grid_rows: int | None = None,
    grid_cols: int | None = None,
    native_backend: str = "auto",
    return_alpha: bool = True,
    return_chol: bool = True,
    workdir: Path | None = None,
    verbose: bool = False,
) -> ScaLAPACKFitResult:
    prepared = prepare_scalapack_fit_workdir(
        K_rr,
        y,
        launcher=launcher,
        nprocs=nprocs,
        executable=executable,
        block_size=block_size,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        native_backend=native_backend,
        return_alpha=return_alpha,
        return_chol=return_chol,
        workdir=workdir,
    )
    return _run_prepared_fit(
        prepared,
        native_backend=native_backend,
        block_size=block_size,
        verbose=verbose,
    )


def fit_exact_gp_scalapack_from_features(
    x_rated: np.ndarray,
    y: np.ndarray,
    *,
    length_scale: float,
    variance: float,
    noise: float,
    launcher: str = "srun",
    nprocs: int = 4,
    executable: str = "native/build/scalapack_gp_fit",
    block_size: int = 128,
    grid_rows: int | None = None,
    grid_cols: int | None = None,
    native_backend: str = "auto",
    return_alpha: bool = True,
    return_chol: bool = True,
    workdir: Path | None = None,
    verbose: bool = False,
) -> ScaLAPACKFitResult:
    # Warn if problem size is too small for ScaLAPACK to be beneficial
    n = x_rated.shape[0]
    _check_problem_size_and_warn(n, nprocs)

    prepared = prepare_scalapack_feature_fit_workdir(
        x_rated,
        y,
        length_scale=length_scale,
        variance=variance,
        noise=noise,
        launcher=launcher,
        nprocs=nprocs,
        executable=executable,
        block_size=block_size,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        native_backend=native_backend,
        return_alpha=return_alpha,
        return_chol=return_chol,
        workdir=workdir,
    )
    return _run_prepared_fit(
        prepared,
        native_backend=native_backend,
        block_size=block_size,
        verbose=verbose,
    )


def fit_exact_gp_scalapack_from_rated(
    x_rated: np.ndarray,
    y_rated: np.ndarray,
    *,
    length_scale: float = 1.0,
    variance: float = 1.0,
    noise: float = 1e-3,
    launcher: str = "srun",
    nprocs: int = 4,
    executable: str = "native/build/scalapack_gp_fit",
    block_size: int = 128,
    grid_rows: int | None = None,
    grid_cols: int | None = None,
    native_backend: str = "auto",
    return_alpha: bool = True,
    return_chol: bool = True,
    workdir: Path | None = None,
    verbose: bool = False,
) -> GPState:
    x_rated = np.asarray(x_rated, dtype=np.float64)
    y_rated = np.asarray(y_rated, dtype=np.float64)
    if x_rated.ndim != 2:
        raise ValueError("x_rated must be 2D")
    if y_rated.ndim != 1:
        raise ValueError("y_rated must be 1D")
    if x_rated.shape[0] != y_rated.shape[0]:
        raise ValueError("x_rated and y_rated length mismatch")
    if noise <= 0:
        raise ValueError("noise must be positive")

    result = fit_exact_gp_scalapack_from_features(
        x_rated,
        y_rated,
        length_scale=length_scale,
        variance=variance,
        noise=noise,
        launcher=launcher,
        nprocs=nprocs,
        executable=executable,
        block_size=block_size,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        native_backend=native_backend,
        return_alpha=return_alpha,
        return_chol=return_chol,
        workdir=workdir,
        verbose=verbose,
    )
    lml = -0.5 * float(y_rated @ result.alpha) - 0.5 * float(result.logdet) - 0.5 * len(y_rated) * np.log(2.0 * np.pi)

    # Only include cho_factor_data if Cholesky was actually returned
    cho_factor_data = None
    if return_chol and np.any(result.chol_lower != 0):
        cho_factor_data = (result.chol_lower, True)

    return GPState(
        x_rated=x_rated,
        y_rated=y_rated,
        alpha=result.alpha,
        cho_factor_data=cho_factor_data,
        length_scale=length_scale,
        variance=variance,
        noise=noise,
        log_marginal_likelihood=float(lml),
        optimization_result={
            "fit_backend": "native_reference",
            "native_backend": result.backend,
            "requested_native_backend": result.requested_backend,
            "compiled_with_scalapack": result.compiled_with_scalapack,
            "fit_total_seconds": result.total_seconds,
            "message": result.message,
            "workdir": str(result.workdir),
            "input_mode": "features",
        },
    )
