"""
Client for persistent ScaLAPACK daemon.

This module manages the lifecycle of the ScaLAPACK daemon and provides
a Python interface for submitting fit requests without subprocess overhead.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


class ScaLAPACKDaemonClient:
    """
    Client for communicating with persistent ScaLAPACK daemon.

    The daemon eliminates ~160ms subprocess overhead by keeping MPI processes
    alive across multiple fit operations.
    """

    def __init__(
        self,
        nprocs: int = 4,
        launcher: str = "mpirun",
        daemon_exe: Path | None = None,
    ):
        self.nprocs = nprocs
        self.launcher = launcher
        self.daemon_exe = daemon_exe or Path("native/build/scalapack_daemon")

        # Create named pipes in temp directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="scalapack_daemon_"))
        self.request_pipe = self.temp_dir / "request.fifo"
        self.response_pipe = self.temp_dir / "response.fifo"

        self.daemon_process: subprocess.Popen | None = None
        self._started = False

    def start(self) -> None:
        """Start the daemon if not already running."""
        if self._started:
            return

        # Create named pipes
        os.mkfifo(self.request_pipe)
        os.mkfifo(self.response_pipe)

        # Launch daemon
        cmd = [
            self.launcher,
            "-n", str(self.nprocs),
            "--bind-to", "none",
            "--map-by", "slot",
            str(self.daemon_exe.resolve()),
            str(self.request_pipe),
            str(self.response_pipe),
        ]

        print(f"[DaemonClient] Starting daemon with {self.nprocs} processes")
        print(f"[DaemonClient] Command: {' '.join(cmd)}")

        self.daemon_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Give daemon time to initialize
        time.sleep(0.5)

        # Check if daemon started successfully
        if self.daemon_process.poll() is not None:
            stdout, stderr = self.daemon_process.communicate()
            raise RuntimeError(
                f"Daemon failed to start:\n"
                f"stdout: {stdout.decode()}\n"
                f"stderr: {stderr.decode()}"
            )

        self._started = True
        print(f"[DaemonClient] Daemon started successfully")

    def shutdown(self) -> None:
        """Shutdown the daemon gracefully."""
        if not self._started or self.daemon_process is None:
            return

        try:
            # Send shutdown request
            request = {"operation": "shutdown"}
            request_json = json.dumps(request) + "\n"

            with open(self.request_pipe, "w") as f:
                f.write(request_json)

            # Read response
            with open(self.response_pipe, "r") as f:
                response_json = f.read()

            # Wait for daemon to exit
            self.daemon_process.wait(timeout=5.0)
            print("[DaemonClient] Daemon shutdown successfully")

        except Exception as e:
            print(f"[DaemonClient] Error during shutdown: {e}")
            if self.daemon_process:
                self.daemon_process.kill()
                self.daemon_process.wait()

        finally:
            # Cleanup pipes and temp directory
            try:
                self.request_pipe.unlink(missing_ok=True)
                self.response_pipe.unlink(missing_ok=True)
                self.temp_dir.rmdir()
            except Exception as e:
                print(f"[DaemonClient] Error cleaning up temp files: {e}")

            self._started = False
            self.daemon_process = None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        length_scale: float,
        variance: float,
        noise: float,
        nprow: int,
        npcol: int,
        block_size: int,
    ) -> dict[str, Any]:
        """
        Submit a fit request to the daemon.

        Returns:
            dict with keys:
                - alpha: np.ndarray
                - L_factor: np.ndarray
                - log_marginal_likelihood: float
                - fit_seconds: float
                - total_seconds: float
        """
        if not self._started:
            raise RuntimeError("Daemon not started. Call start() first.")

        m, d = x.shape
        if y.shape != (m,):
            raise ValueError(f"y shape {y.shape} does not match x shape {x.shape}")

        # Write input data to temp files
        x_path = self.temp_dir / "x_input.bin"
        y_path = self.temp_dir / "y_input.bin"
        alpha_path = self.temp_dir / "alpha_output.bin"
        L_path = self.temp_dir / "L_output.bin"

        x.astype(np.float64, copy=False).tofile(x_path)
        y.astype(np.float64, copy=False).tofile(y_path)

        # Create request
        request = {
            "operation": "fit",
            "m": int(m),
            "d": int(d),
            "length_scale": float(length_scale),
            "variance": float(variance),
            "noise": float(noise),
            "nprow": int(nprow),
            "npcol": int(npcol),
            "block_size": int(block_size),
            "x_path": str(x_path),
            "y_path": str(y_path),
            "alpha_out_path": str(alpha_path),
            "L_out_path": str(L_path),
        }

        request_json = json.dumps(request) + "\n"

        # Send request
        with open(self.request_pipe, "w") as f:
            f.write(request_json)

        # Read response
        with open(self.response_pipe, "r") as f:
            response_json = f.read()

        response = json.loads(response_json)

        if response["status"] != 0:
            raise RuntimeError(f"Daemon error: {response['message']}")

        # Read output data
        alpha = np.fromfile(alpha_path, dtype=np.float64)
        L_factor = np.fromfile(L_path, dtype=np.float64).reshape(m, m)

        # Cleanup temp data files
        x_path.unlink()
        y_path.unlink()
        alpha_path.unlink()
        L_path.unlink()

        return {
            "alpha": alpha,
            "L_factor": L_factor,
            "log_marginal_likelihood": response["log_marginal_likelihood"],
            "fit_seconds": response["fit_seconds"],
            "total_seconds": response["total_seconds"],
        }

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False


# Global daemon instance for reuse
_global_daemon: ScaLAPACKDaemonClient | None = None


def get_global_daemon(nprocs: int = 4, launcher: str = "mpirun") -> ScaLAPACKDaemonClient:
    """
    Get or create the global daemon instance.

    This allows reusing the same daemon across multiple operations without
    manually managing the lifecycle.
    """
    global _global_daemon

    if _global_daemon is None:
        _global_daemon = ScaLAPACKDaemonClient(nprocs=nprocs, launcher=launcher)
        _global_daemon.start()
    elif _global_daemon.nprocs != nprocs:
        # Different number of processes requested, restart
        _global_daemon.shutdown()
        _global_daemon = ScaLAPACKDaemonClient(nprocs=nprocs, launcher=launcher)
        _global_daemon.start()

    return _global_daemon


def shutdown_global_daemon() -> None:
    """Shutdown the global daemon if running."""
    global _global_daemon
    if _global_daemon is not None:
        _global_daemon.shutdown()
        _global_daemon = None
