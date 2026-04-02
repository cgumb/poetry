from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..gp_exact import GPState
from ..kernel import rbf_kernel


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

    if launcher == "srun":
        command = [launcher, "-n", str(nprocs), executable]
    elif launcher == "mpirun":
        command = [launcher, "-np", str(nprocs), executable]
    else:
        raise ValueError("launcher must be 'srun' or 'mpirun'")
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
    workdir: Path | None = None,
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
        workdir=workdir,
    )
    subprocess.run(prepared.command, check=True)
    meta = json.loads(prepared.output_meta_path.read_text())
    n = int(meta["n"])
    alpha = np.fromfile(prepared.alpha_bin_path, dtype=np.float64, count=n)
    chol = np.fromfile(prepared.chol_bin_path, dtype=np.float64, count=n * n).reshape(n, n)
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
        message=str(meta.get("message", "")),
        workdir=prepared.workdir,
    )
    if result.info_potrf != 0 or result.info_potrs not in {0, -1}:
        raise RuntimeError(result.message)
    if not result.implemented:
        raise NotImplementedError(result.message)
    return result


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
    workdir: Path | None = None,
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

    K_rr = rbf_kernel(x_rated, x_rated, length_scale=length_scale, variance=variance)
    K_rr.flat[:: K_rr.shape[0] + 1] += noise * noise
    result = fit_exact_gp_scalapack(
        K_rr,
        y_rated,
        launcher=launcher,
        nprocs=nprocs,
        executable=executable,
        block_size=block_size,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        workdir=workdir,
    )
    lml = -0.5 * float(y_rated @ result.alpha) - 0.5 * float(result.logdet) - 0.5 * len(y_rated) * np.log(2.0 * np.pi)
    return GPState(
        x_rated=x_rated,
        y_rated=y_rated,
        alpha=result.alpha,
        cho_factor_data=(result.chol_lower, True),
        length_scale=length_scale,
        variance=variance,
        noise=noise,
        log_marginal_likelihood=float(lml),
        optimization_result={
            "fit_backend": "native_reference",
            "native_backend": result.backend,
            "fit_total_seconds": result.total_seconds,
            "message": result.message,
        },
    )
