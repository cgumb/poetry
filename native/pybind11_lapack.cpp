/**
 * PyBind11 LAPACK Bridge for Poetry GP
 *
 * In-memory bridge between Python and LAPACK, eliminating subprocess
 * and file I/O overhead (~1.5-2.5s per fit).
 *
 * Phase 1: Single-process LAPACK (no MPI)
 * - Suitable for m < 5000
 * - Zero overhead compared to scipy
 * - Multi-threaded BLAS support via OpenMP
 *
 * Phase 2 (future): MPI daemon with shared memory
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace py = pybind11;

// LAPACK/BLAS function declarations
extern "C" {
    // Cholesky factorization: A = L * L^T
    void dpotrf_(const char* uplo, const int* n, double* a, const int* lda, int* info);

    // Solve A * X = B using Cholesky factorization
    void dpotrs_(const char* uplo, const int* n, const int* nrhs,
                 const double* a, const int* lda, double* b, const int* ldb, int* info);

    // Matrix-matrix multiply: C = alpha * A * B + beta * C
    void dgemm_(const char* transa, const char* transb,
                const int* m, const int* n, const int* k,
                const double* alpha, const double* a, const int* lda,
                const double* b, const int* ldb,
                const double* beta, double* c, const int* ldc);

    // Triangular solve: B = alpha * op(A)^{-1} * B
    void dtrsm_(const char* side, const char* uplo, const char* transa, const char* diag,
                const int* m, const int* n, const double* alpha,
                const double* a, const int* lda, double* b, const int* ldb);
}


/**
 * Compute RBF kernel: K(x, y) = variance * exp(-||x - y||^2 / (2 * length_scale^2))
 *
 * Args:
 *   x1: (n1 × d) matrix
 *   x2: (n2 × d) matrix
 *   length_scale: RBF length scale
 *   variance: RBF variance (output scale)
 *
 * Returns:
 *   K: (n1 × n2) kernel matrix
 */
py::array_t<double> rbf_kernel(
    py::array_t<double, py::array::c_style | py::array::forcecast> x1_py,
    py::array_t<double, py::array::c_style | py::array::forcecast> x2_py,
    double length_scale,
    double variance
) {
    auto x1_buf = x1_py.request();
    auto x2_buf = x2_py.request();

    if (x1_buf.ndim != 2 || x2_buf.ndim != 2)
        throw std::runtime_error("x1 and x2 must be 2D arrays");
    if (x1_buf.shape[1] != x2_buf.shape[1])
        throw std::runtime_error("x1 and x2 must have same number of columns");

    int n1 = static_cast<int>(x1_buf.shape[0]);
    int n2 = static_cast<int>(x2_buf.shape[0]);
    int d = static_cast<int>(x1_buf.shape[1]);

    const double* x1 = static_cast<const double*>(x1_buf.ptr);
    const double* x2 = static_cast<const double*>(x2_buf.ptr);

    // Allocate output
    auto K_py = py::array_t<double>({n1, n2});
    auto K_buf = K_py.request();
    double* K = static_cast<double*>(K_buf.ptr);

    // Compute pairwise squared distances
    const double inv_2l2 = -0.5 / (length_scale * length_scale);

    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            double dist_sq = 0.0;
            for (int k = 0; k < d; ++k) {
                double diff = x1[i * d + k] - x2[j * d + k];
                dist_sq += diff * diff;
            }
            K[i * n2 + j] = variance * std::exp(inv_2l2 * dist_sq);
        }
    }

    return K_py;
}


/**
 * Fit GP using LAPACK Cholesky factorization.
 *
 * Computes:
 *   L = cholesky(K_rr)  [lower Cholesky: K = L * L^T]
 *   alpha = L^{-T} * L^{-1} * y
 *   logdet = 2 * sum(log(diag(L)))
 *
 * Args:
 *   K_rr: (m × m) kernel matrix (any order - will be converted internally)
 *   y: (m,) observation vector
 *   return_chol: Whether to return Cholesky factor (for variance computation)
 *
 * Returns:
 *   dict with keys:
 *     - alpha: (m,) solution vector
 *     - chol_lower: (m × m) lower Cholesky factor in F-order (only if return_chol=True)
 *     - logdet: scalar log-determinant
 *     - info_potrf: LAPACK info code for Cholesky
 *     - info_potrs: LAPACK info code for solve
 */
py::dict fit_gp_lapack(
    py::array_t<double> K_rr_py,  // Accept any order
    py::array_t<double> y_py,
    bool return_chol = true
) {
    auto K_rr_buf = K_rr_py.request();
    auto y_buf = y_py.request();

    if (K_rr_buf.ndim != 2 || K_rr_buf.shape[0] != K_rr_buf.shape[1])
        throw std::runtime_error("K_rr must be square 2D array");
    if (y_buf.ndim != 1 || y_buf.shape[0] != K_rr_buf.shape[0])
        throw std::runtime_error("y must be 1D array matching K_rr");

    int m = static_cast<int>(K_rr_buf.shape[0]);

    // Convert K_rr to Fortran-order (column-major) for LAPACK
    // LAPACK expects column-major layout
    const double* K_rr = static_cast<const double*>(K_rr_buf.ptr);
    std::vector<double> chol(m * m);

    // Check if input is already Fortran-order
    bool is_f_order = (K_rr_buf.strides[0] == sizeof(double) &&
                       K_rr_buf.strides[1] == m * sizeof(double));

    if (is_f_order) {
        // Already column-major, just copy
        std::memcpy(chol.data(), K_rr, m * m * sizeof(double));
    } else {
        // C-order (row-major): transpose during copy to get column-major
        // C-order element (i,j) at position i*m + j
        // F-order element (i,j) at position j*m + i
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                chol[j * m + i] = K_rr[i * m + j];
            }
        }
    }

    std::vector<double> alpha(m);
    std::memcpy(alpha.data(), y_buf.ptr, m * sizeof(double));

    // Cholesky factorization: A = L * L^T (lower triangle)
    const char uplo = 'L';
    int info_potrf = 0;
    dpotrf_(&uplo, &m, chol.data(), &m, &info_potrf);

    if (info_potrf != 0) {
        throw std::runtime_error(
            "Cholesky factorization failed with info=" + std::to_string(info_potrf) +
            ". Matrix may not be positive definite."
        );
    }

    // Compute logdet: 2 * sum(log(diag(L)))
    double logdet = 0.0;
    for (int i = 0; i < m; ++i) {
        // Diagonal element (i,i) at position i*m + i in column-major
        double diag_val = chol[i * m + i];
        if (diag_val <= 0.0) {
            throw std::runtime_error("Non-positive diagonal in Cholesky factor");
        }
        logdet += 2.0 * std::log(diag_val);
    }

    // Solve for alpha: K_rr * alpha = y using lower Cholesky
    int nrhs = 1;
    int info_potrs = 0;
    dpotrs_(&uplo, &m, &nrhs, chol.data(), &m, alpha.data(), &m, &info_potrs);

    if (info_potrs != 0) {
        throw std::runtime_error(
            "Linear solve failed with info=" + std::to_string(info_potrs)
        );
    }

    // Build result dictionary with Python-owned arrays
    py::dict result;

    // Copy alpha to Python-owned array
    auto alpha_py = py::array_t<double>(m);
    auto alpha_buf = alpha_py.mutable_unchecked<1>();
    for (int i = 0; i < m; ++i) {
        alpha_buf(i) = alpha[i];
    }
    result["alpha"] = alpha_py;

    result["logdet"] = logdet;
    result["info_potrf"] = info_potrf;
    result["info_potrs"] = info_potrs;

    if (return_chol) {
        // Zero out upper triangle (LAPACK only fills lower with uplo='L')
        for (int i = 0; i < m; ++i) {
            for (int j = i + 1; j < m; ++j) {
                chol[j * m + i] = 0.0;  // Column-major: element (i,j) at j*m+i
            }
        }

        // Return lower Cholesky in Fortran-order (column-major)
        // chol is already in column-major format
        auto chol_py = py::array_t<double>(
            {m, m},                                    // shape
            {sizeof(double), m * sizeof(double)}      // strides: column-major
        );
        auto chol_buf = chol_py.mutable_unchecked<2>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                chol_buf(i, j) = chol[j * m + i];  // Read from column-major storage
            }
        }
        result["chol_lower"] = chol_py;
    }

    return result;
}


/**
 * Predict GP posterior mean and variance using LAPACK triangular solve.
 *
 * Computes:
 *   mean = K_qr @ alpha
 *   variance = variance - sum((L^{-1} @ K_qr.T)^2, axis=0)
 *
 * Args:
 *   x_query: (n × d) query points
 *   x_rated: (m × d) training points
 *   alpha: (m,) fitted GP coefficients
 *   chol_lower: (m × m) lower Cholesky factor (only needed for variance)
 *   length_scale: RBF length scale
 *   variance: RBF variance
 *   compute_variance: Whether to compute variance (expensive O(n × m^2))
 *
 * Returns:
 *   dict with keys:
 *     - mean: (n,) posterior mean
 *     - variance: (n,) posterior variance (only if compute_variance=True)
 */
py::dict predict_gp_lapack(
    py::array_t<double, py::array::c_style | py::array::forcecast> x_query_py,
    py::array_t<double, py::array::c_style | py::array::forcecast> x_rated_py,
    py::array_t<double, py::array::c_style | py::array::forcecast> alpha_py,
    py::array_t<double> chol_lower_py,  // Accept any order (will be Fortran from fit)
    double length_scale,
    double variance,
    bool compute_variance = true
) {
    auto x_query_buf = x_query_py.request();
    auto x_rated_buf = x_rated_py.request();
    auto alpha_buf = alpha_py.request();

    if (x_query_buf.ndim != 2 || x_rated_buf.ndim != 2)
        throw std::runtime_error("x_query and x_rated must be 2D");
    if (x_query_buf.shape[1] != x_rated_buf.shape[1])
        throw std::runtime_error("x_query and x_rated must have same number of features");
    if (alpha_buf.ndim != 1 || alpha_buf.shape[0] != x_rated_buf.shape[0])
        throw std::runtime_error("alpha must be 1D with length matching x_rated");

    int n = static_cast<int>(x_query_buf.shape[0]);
    int m = static_cast<int>(x_rated_buf.shape[0]);
    int d = static_cast<int>(x_query_buf.shape[1]);

    // Compute K_qr = rbf_kernel(x_query, x_rated)
    auto K_qr_py = rbf_kernel(x_query_py, x_rated_py, length_scale, variance);
    auto K_qr_buf = K_qr_py.request();
    const double* K_qr = static_cast<const double*>(K_qr_buf.ptr);
    const double* alpha = static_cast<const double*>(alpha_buf.ptr);

    // Compute mean: K_qr @ alpha
    std::vector<double> mean(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            mean[i] += K_qr[i * m + j] * alpha[j];
        }
    }

    // Copy mean to Python-owned array
    auto mean_py = py::array_t<double>(n);
    auto mean_buf = mean_py.mutable_unchecked<1>();
    for (int i = 0; i < n; ++i) {
        mean_buf(i) = mean[i];
    }

    py::dict result;
    result["mean"] = mean_py;

    if (!compute_variance) {
        return result;
    }

    // Variance computation: expensive O(n × m^2)
    auto chol_buf = chol_lower_py.request();
    if (chol_buf.ndim != 2 || chol_buf.shape[0] != m || chol_buf.shape[1] != m)
        throw std::runtime_error("chol_lower must be (m × m)");

    // Copy K_qr^T for triangular solve (LAPACK modifies in-place)
    std::vector<double> K_qr_T(m * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            K_qr_T[j * n + i] = K_qr[i * m + j];
        }
    }

    // Solve: v = L^{-1} @ K_qr^T (where L is lower Cholesky factor)
    // Using dtrsm: B = alpha * op(A)^{-1} * B
    // Note: chol_lower_py contains L (lower), not U, because fit converts U -> L via transpose
    const char side = 'L';    // A is on the left
    const char uplo = 'L';    // A is lower triangular
    const char transa = 'N';  // No transpose
    const char diag = 'N';    // Non-unit diagonal
    const double alpha_val = 1.0;
    int m_int = m;
    int n_int = n;

    const double* chol = static_cast<const double*>(chol_buf.ptr);
    std::vector<double> chol_copy(m * m);
    std::memcpy(chol_copy.data(), chol, m * m * sizeof(double));

    dtrsm_(&side, &uplo, &transa, &diag, &m_int, &n_int, &alpha_val,
           chol_copy.data(), &m_int, K_qr_T.data(), &m_int);

    // Compute variance: variance - sum(v^2, axis=0)
    std::vector<double> var(n);
    for (int i = 0; i < n; ++i) {
        double sum_sq = 0.0;
        for (int j = 0; j < m; ++j) {
            double val = K_qr_T[j * n + i];
            sum_sq += val * val;
        }
        var[i] = std::max(0.0, variance - sum_sq);  // Clamp to non-negative
    }

    // Copy variance to Python-owned array
    auto var_py = py::array_t<double>(n);
    auto var_buf = var_py.mutable_unchecked<1>();
    for (int i = 0; i < n; ++i) {
        var_buf(i) = var[i];
    }

    result["variance"] = var_py;
    return result;
}


PYBIND11_MODULE(poetry_gp_native, m) {
    m.doc() = R"pbdoc(
        Poetry GP Native Module (PyBind11 LAPACK Bridge)

        In-memory bridge for GP fitting and prediction using LAPACK.
        Eliminates subprocess and file I/O overhead (~1.5-2.5s per fit).

        Suitable for:
        - Interactive CLI sessions (fast iteration)
        - Small-to-medium problems (m < 5000)
        - Single-node workloads

        Functions:
        - rbf_kernel: Compute RBF kernel matrix
        - fit_gp_lapack: Fit GP via Cholesky factorization
        - predict_gp_lapack: Predict mean/variance via triangular solve
    )pbdoc";

    m.def("rbf_kernel", &rbf_kernel,
          R"pbdoc(
              Compute RBF kernel matrix.

              Args:
                  x1: (n1 × d) input points
                  x2: (n2 × d) input points
                  length_scale: RBF length scale
                  variance: RBF variance (output scale)

              Returns:
                  K: (n1 × n2) kernel matrix
          )pbdoc",
          py::arg("x1"), py::arg("x2"),
          py::arg("length_scale"), py::arg("variance"));

    m.def("fit_gp_lapack", &fit_gp_lapack,
          R"pbdoc(
              Fit GP using LAPACK Cholesky factorization.

              Args:
                  K_rr: (m × m) kernel matrix
                  y: (m,) observations
                  return_chol: Whether to return Cholesky factor (default: True)

              Returns:
                  dict with alpha, logdet, info codes, and optionally chol_lower

              Complexity: O(m³) for Cholesky, O(m²) for solve
          )pbdoc",
          py::arg("K_rr"), py::arg("y"),
          py::arg("return_chol") = true);

    m.def("predict_gp_lapack", &predict_gp_lapack,
          R"pbdoc(
              Predict GP posterior mean and variance.

              Args:
                  x_query: (n × d) query points
                  x_rated: (m × d) training points
                  alpha: (m,) fitted coefficients
                  chol_lower: (m × m) Cholesky factor
                  length_scale: RBF length scale
                  variance: RBF variance
                  compute_variance: Compute variance (default: True)

              Returns:
                  dict with mean and optionally variance

              Complexity:
                  Mean only: O(n × m × d) + O(n × m)
                  With variance: O(n × m × d) + O(n × m²)
          )pbdoc",
          py::arg("x_query"), py::arg("x_rated"), py::arg("alpha"),
          py::arg("chol_lower"), py::arg("length_scale"), py::arg("variance"),
          py::arg("compute_variance") = true);
}
