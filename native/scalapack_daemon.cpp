/*
 * Persistent ScaLAPACK daemon for GP fitting.
 *
 * This daemon keeps MPI processes alive across multiple fit operations,
 * eliminating the ~160ms subprocess overhead per operation.
 *
 * Communication via named pipes (FIFOs):
 * - Request pipe: client -> daemon
 * - Response pipe: daemon -> client
 *
 * Protocol: JSON over pipes
 */

#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <signal.h>

// BLAS functions for kernel computation
extern "C" {
void dgemm_(const char* transa, const char* transb,
            const int* m, const int* n, const int* k,
            const double* alpha, const double* a, const int* lda,
            const double* b, const int* ldb,
            const double* beta, double* c, const int* ldc);

void dtrsm_(const char* side, const char* uplo, const char* transa, const char* diag,
            const int* m, const int* n, const double* alpha,
            const double* a, const int* lda, double* b, const int* ldb);

void dgemv_(const char* trans, const int* m, const int* n,
            const double* alpha, const double* a, const int* lda,
            const double* x, const int* incx,
            const double* beta, double* y, const int* incy);
}

// Compute RBF kernel between query and rated points
// k_qr[i,j] = variance * exp(-0.5 * ||x_query[i] - x_rated[j]||^2 / length_scale^2)
void compute_rbf_kernel(
    const double* x_query, int n_query,
    const double* x_rated, int n_rated,
    int d,
    double length_scale, double variance,
    double* k_qr  // Output: n_query × n_rated
) {
    const double inv_two_ell_sq = -0.5 / (length_scale * length_scale);

    // Compute squared norms
    std::vector<double> query_norms(n_query, 0.0);
    std::vector<double> rated_norms(n_rated, 0.0);

    for (int i = 0; i < n_query; ++i) {
        for (int k = 0; k < d; ++k) {
            double val = x_query[i * d + k];
            query_norms[i] += val * val;
        }
    }

    for (int j = 0; j < n_rated; ++j) {
        for (int k = 0; k < d; ++k) {
            double val = x_rated[j * d + k];
            rated_norms[j] += val * val;
        }
    }

    // Compute Gram matrix using DGEMM: gram = x_query @ x_rated.T
    // Both matrices are row-major, so we use the transpose trick
    const char transa = 'T';
    const char transb = 'N';
    const double alpha_blas = 1.0;
    const double beta_blas = 0.0;
    const int m = n_query;
    const int n = n_rated;
    const int k = d;
    const int lda = d;
    const int ldb = d;
    const int ldc = n_query;

    dgemm_(&transa, &transb, &m, &n, &k, &alpha_blas,
           x_query, &lda, x_rated, &ldb, &beta_blas, k_qr, &ldc);

    // Apply RBF transformation: k[i,j] = variance * exp(-0.5 * ||x_i - x_j||^2 / ell^2)
    for (int i = 0; i < n_query; ++i) {
        for (int j = 0; j < n_rated; ++j) {
            double dot = k_qr[i + j * n_query];  // Column-major from DGEMM
            double d2 = query_norms[i] + rated_norms[j] - 2.0 * dot;
            if (d2 < 0.0) d2 = 0.0;
            k_qr[i + j * n_query] = variance * std::exp(inv_two_ell_sq * d2);
        }
    }
}

// Forward declaration from scalapack_gp_fit.cpp
struct NativeResult {
    bool implemented = false;
    std::string backend = "unknown";
    std::string requested_backend = "auto";
    std::string message;
    int info_potrf = -1;
    int info_potrs = -1;
    double factor_seconds = 0.0;
    double solve_seconds = 0.0;
    double gather_seconds = 0.0;
    double total_seconds = 0.0;
    double logdet = 0.0;
    std::vector<double> alpha;
    std::vector<double> chol;
    bool has_alpha = false;
    bool has_chol = false;
};

#ifdef HAVE_SCALAPACK
NativeResult run_scalapack_distributed(
    std::size_t n,
    int rank,
    int size,
    const std::vector<double>& x_rated_root,
    const std::vector<double>& rhs_root,
    int d,
    double length_scale,
    double variance,
    double noise,
    int block_size,
    bool return_alpha,
    bool return_chol,
    MPI_Comm comm
);
#endif

volatile sig_atomic_t shutdown_requested = 0;

void signal_handler(int signum) {
    shutdown_requested = 1;
}

struct DaemonRequest {
    std::string operation;  // "fit", "score", "shutdown"

    // Fit parameters
    int m;
    int d;
    double length_scale;
    double variance;
    double noise;
    int block_size;
    std::string x_path;
    std::string y_path;
    std::string alpha_out_path;
    std::string L_out_path;

    // Score parameters
    int n_query;  // Number of query points to score
    std::string x_query_path;  // Query points (n_query × d)
    std::string x_rated_path;  // Rated points (m × d)
    std::string alpha_path;    // Alpha vector (m)
    std::string L_path;        // Cholesky factor (m × m)
    std::string mean_out_path;  // Output mean (n_query)
    std::string var_out_path;   // Output variance (n_query)
};

struct DaemonResponse {
    int status;  // 0=success, 1=error
    std::string message;
    double log_marginal_likelihood;
    double fit_seconds;
    double total_seconds;
};

// Simple JSON parser (manual for minimal dependencies)
bool parse_request_json(const std::string& json, DaemonRequest& req) {
    // Very simple parser - handles spaces after colons
    // Real implementation would use a proper JSON library

    fprintf(stderr, "[Daemon] Parsing JSON: %s\n", json.substr(0, 200).c_str());

    // Find operation - handle optional space after colon
    size_t op_pos = json.find("\"operation\"");
    if (op_pos == std::string::npos) {
        fprintf(stderr, "[Daemon] Could not find 'operation' field\n");
        return false;
    }
    size_t colon_pos = json.find(":", op_pos);
    if (colon_pos == std::string::npos) return false;
    size_t op_start = json.find("\"", colon_pos + 1);
    if (op_start == std::string::npos) return false;
    op_start++;  // Skip opening quote
    size_t op_end = json.find("\"", op_start);
    if (op_end == std::string::npos) return false;
    req.operation = json.substr(op_start, op_end - op_start);

    if (req.operation == "shutdown") {
        return true;
    }

    // Parse integers - handle optional space after colon
    auto parse_int = [&](const char* key, int& val) {
        std::string key_str = std::string("\"") + key + "\"";
        size_t pos = json.find(key_str);
        if (pos == std::string::npos) return false;
        size_t colon = json.find(":", pos);
        if (colon == std::string::npos) return false;
        val = std::atoi(json.c_str() + colon + 1);
        return true;
    };

    // Parse doubles - handle optional space after colon
    auto parse_double = [&](const char* key, double& val) {
        std::string key_str = std::string("\"") + key + "\"";
        size_t pos = json.find(key_str);
        if (pos == std::string::npos) return false;
        size_t colon = json.find(":", pos);
        if (colon == std::string::npos) return false;
        val = std::atof(json.c_str() + colon + 1);
        return true;
    };

    // Parse strings - handle optional space after colon
    auto parse_string = [&](const char* key, std::string& val) {
        std::string key_str = std::string("\"") + key + "\"";
        size_t pos = json.find(key_str);
        if (pos == std::string::npos) return false;
        size_t colon = json.find(":", pos);
        if (colon == std::string::npos) return false;
        size_t start = json.find("\"", colon + 1);
        if (start == std::string::npos) return false;
        start++;  // Skip opening quote
        size_t end = json.find("\"", start);
        if (end == std::string::npos) return false;
        val = json.substr(start, end - start);
        return true;
    };

    // Parse operation-specific parameters
    if (req.operation == "score") {
        // Score operation parameters
        if (!parse_int("n_query", req.n_query)) {
            fprintf(stderr, "[Daemon] Failed to parse 'n_query'\n");
            return false;
        }
        if (!parse_int("m", req.m)) {
            fprintf(stderr, "[Daemon] Failed to parse 'm'\n");
            return false;
        }
        if (!parse_int("d", req.d)) {
            fprintf(stderr, "[Daemon] Failed to parse 'd'\n");
            return false;
        }
        if (!parse_double("length_scale", req.length_scale)) {
            fprintf(stderr, "[Daemon] Failed to parse 'length_scale'\n");
            return false;
        }
        if (!parse_double("variance", req.variance)) {
            fprintf(stderr, "[Daemon] Failed to parse 'variance'\n");
            return false;
        }
        if (!parse_string("x_query_path", req.x_query_path)) {
            fprintf(stderr, "[Daemon] Failed to parse 'x_query_path'\n");
            return false;
        }
        if (!parse_string("x_rated_path", req.x_rated_path)) {
            fprintf(stderr, "[Daemon] Failed to parse 'x_rated_path'\n");
            return false;
        }
        if (!parse_string("alpha_path", req.alpha_path)) {
            fprintf(stderr, "[Daemon] Failed to parse 'alpha_path'\n");
            return false;
        }
        if (!parse_string("L_path", req.L_path)) {
            fprintf(stderr, "[Daemon] Failed to parse 'L_path'\n");
            return false;
        }
        if (!parse_string("mean_out_path", req.mean_out_path)) {
            fprintf(stderr, "[Daemon] Failed to parse 'mean_out_path'\n");
            return false;
        }
        if (!parse_string("var_out_path", req.var_out_path)) {
            fprintf(stderr, "[Daemon] Failed to parse 'var_out_path'\n");
            return false;
        }
        return true;
    }

    // Fit operation parameters
    if (!parse_int("m", req.m)) {
        fprintf(stderr, "[Daemon] Failed to parse 'm'\n");
        return false;
    }
    if (!parse_int("d", req.d)) {
        fprintf(stderr, "[Daemon] Failed to parse 'd'\n");
        return false;
    }
    if (!parse_double("length_scale", req.length_scale)) {
        fprintf(stderr, "[Daemon] Failed to parse 'length_scale'\n");
        return false;
    }
    if (!parse_double("variance", req.variance)) {
        fprintf(stderr, "[Daemon] Failed to parse 'variance'\n");
        return false;
    }
    if (!parse_double("noise", req.noise)) {
        fprintf(stderr, "[Daemon] Failed to parse 'noise'\n");
        return false;
    }
    if (!parse_int("block_size", req.block_size)) {
        fprintf(stderr, "[Daemon] Failed to parse 'block_size'\n");
        return false;
    }
    if (!parse_string("x_path", req.x_path)) {
        fprintf(stderr, "[Daemon] Failed to parse 'x_path'\n");
        return false;
    }
    if (!parse_string("y_path", req.y_path)) {
        fprintf(stderr, "[Daemon] Failed to parse 'y_path'\n");
        return false;
    }
    if (!parse_string("alpha_out_path", req.alpha_out_path)) {
        fprintf(stderr, "[Daemon] Failed to parse 'alpha_out_path'\n");
        return false;
    }
    if (!parse_string("L_out_path", req.L_out_path)) {
        fprintf(stderr, "[Daemon] Failed to parse 'L_out_path'\n");
        return false;
    }

    return true;
}

std::string create_response_json(const DaemonResponse& resp) {
    char buf[1024];
    snprintf(buf, sizeof(buf),
        "{\"status\":%d,\"message\":\"%s\",\"log_marginal_likelihood\":%.15e,\"fit_seconds\":%.6f,\"total_seconds\":%.6f}\n",
        resp.status,
        resp.message.c_str(),
        resp.log_marginal_likelihood,
        resp.fit_seconds,
        resp.total_seconds
    );
    return std::string(buf);
}

bool read_binary_file(const std::string& path, std::vector<double>& data, size_t expected_size) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s: %s\n", path.c_str(), strerror(errno));
        return false;
    }

    data.resize(expected_size);
    size_t n = fread(data.data(), sizeof(double), expected_size, f);
    fclose(f);

    if (n != expected_size) {
        fprintf(stderr, "Read %zu elements, expected %zu from %s\n", n, expected_size, path.c_str());
        return false;
    }

    return true;
}

bool write_binary_file(const std::string& path, const double* data, size_t size) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "Failed to open %s for writing: %s\n", path.c_str(), strerror(errno));
        return false;
    }

    size_t n = fwrite(data, sizeof(double), size, f);
    fclose(f);

    if (n != size) {
        fprintf(stderr, "Wrote %zu elements, expected %zu to %s\n", n, size, path.c_str());
        return false;
    }

    return true;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <request_pipe> <response_pipe>\n", argv[0]);
        return 1;
    }

    const char* request_pipe = argv[1];
    const char* response_pipe = argv[2];

    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Setup signal handling (rank 0 only)
    if (rank == 0) {
        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);

        fprintf(stderr, "[Daemon] Started with %d processes\n", size);
        fprintf(stderr, "[Daemon] Request pipe: %s\n", request_pipe);
        fprintf(stderr, "[Daemon] Response pipe: %s\n", response_pipe);
    }

    // Main daemon loop (rank 0 handles I/O)
    while (!shutdown_requested) {
        if (rank == 0) {
            // Open request pipe (blocking until client writes)
            int req_fd = open(request_pipe, O_RDONLY);
            if (req_fd < 0) {
                fprintf(stderr, "[Daemon] Failed to open request pipe: %s\n", strerror(errno));
                break;
            }

            // Read request (up to 4KB)
            char buffer[4096];
            ssize_t n = read(req_fd, buffer, sizeof(buffer) - 1);
            close(req_fd);

            if (n <= 0) {
                fprintf(stderr, "[Daemon] Empty request, shutting down\n");
                shutdown_requested = 1;
                break;
            }

            buffer[n] = '\0';
            std::string request_json(buffer);

            fprintf(stderr, "[Daemon] Received request: %s\n", request_json.c_str());

            // Parse request
            DaemonRequest req;
            if (!parse_request_json(request_json, req)) {
                fprintf(stderr, "[Daemon] Failed to parse request JSON\n");

                DaemonResponse resp;
                resp.status = 1;
                resp.message = "Failed to parse request";
                resp.log_marginal_likelihood = 0.0;
                resp.fit_seconds = 0.0;
                resp.total_seconds = 0.0;

                int resp_fd = open(response_pipe, O_WRONLY);
                if (resp_fd >= 0) {
                    std::string resp_json = create_response_json(resp);
                    write(resp_fd, resp_json.c_str(), resp_json.length());
                    close(resp_fd);
                }
                continue;
            }

            // Handle shutdown
            if (req.operation == "shutdown") {
                fprintf(stderr, "[Daemon] Shutdown requested\n");
                shutdown_requested = 1;

                // Broadcast shutdown op code (0) to all ranks so they can exit cleanly
                int op_code = 0;
                MPI_Bcast(&op_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

                DaemonResponse resp;
                resp.status = 0;
                resp.message = "Shutting down";
                resp.log_marginal_likelihood = 0.0;
                resp.fit_seconds = 0.0;
                resp.total_seconds = 0.0;

                int resp_fd = open(response_pipe, O_WRONLY);
                if (resp_fd >= 0) {
                    std::string resp_json = create_response_json(resp);
                    write(resp_fd, resp_json.c_str(), resp_json.length());
                    close(resp_fd);
                }
                break;
            }

            // Handle score operation (parallel scoring across ranks)
            if (req.operation == "score") {
                fprintf(stderr, "[Daemon] Score operation requested: n_query=%d, m=%d, d=%d\n",
                        req.n_query, req.m, req.d);

                // Broadcast operation code (2 = score)
                int op_code = 2;
                MPI_Bcast(&op_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

                // Broadcast parameters to all ranks
                MPI_Bcast(&req.n_query, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&req.m, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&req.d, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&req.length_scale, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(&req.variance, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                // Load GP state (rank 0 only)
                std::vector<double> x_rated, alpha, L_factor;
                if (!read_binary_file(req.x_rated_path, x_rated, req.m * req.d)) {
                    fprintf(stderr, "[Daemon] Failed to read x_rated\n");
                    continue;
                }
                if (!read_binary_file(req.alpha_path, alpha, req.m)) {
                    fprintf(stderr, "[Daemon] Failed to read alpha\n");
                    continue;
                }
                if (!read_binary_file(req.L_path, L_factor, req.m * req.m)) {
                    fprintf(stderr, "[Daemon] Failed to read L_factor\n");
                    continue;
                }

                // Load query points (rank 0 only)
                std::vector<double> x_query;
                if (!read_binary_file(req.x_query_path, x_query, req.n_query * req.d)) {
                    fprintf(stderr, "[Daemon] Failed to read x_query\n");
                    continue;
                }

                // Broadcast GP state to all ranks
                MPI_Bcast(x_rated.data(), req.m * req.d, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(alpha.data(), req.m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(L_factor.data(), req.m * req.m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                // Distribute query points across ranks
                int fit_rank, fit_size;
                MPI_Comm_rank(MPI_COMM_WORLD, &fit_rank);
                MPI_Comm_size(MPI_COMM_WORLD, &fit_size);

                int chunk_size = (req.n_query + fit_size - 1) / fit_size;
                int my_start = fit_rank * chunk_size;
                int my_end = std::min(my_start + chunk_size, req.n_query);
                int my_n = std::max(0, my_end - my_start);

                fprintf(stderr, "[Daemon rank %d] Scoring chunk: [%d, %d) (%d points)\n",
                        fit_rank, my_start, my_end, my_n);

                // Each rank gets its chunk
                std::vector<double> my_x_query(my_n * req.d);
                std::vector<int> sendcounts(fit_size);
                std::vector<int> displs(fit_size);
                for (int r = 0; r < fit_size; ++r) {
                    int start = r * chunk_size;
                    int end = std::min(start + chunk_size, req.n_query);
                    sendcounts[r] = std::max(0, end - start) * req.d;
                    displs[r] = start * req.d;
                }

                MPI_Scatterv(x_query.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                             my_x_query.data(), my_n * req.d, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                // Compute kernel matrix k_qr for my chunk
                std::vector<double> k_qr(my_n * req.m);
                if (my_n > 0) {
                    compute_rbf_kernel(my_x_query.data(), my_n, x_rated.data(), req.m, req.d,
                                      req.length_scale, req.variance, k_qr.data());
                }

                // Compute mean: k_qr @ alpha (column-major k_qr from compute_rbf_kernel)
                std::vector<double> my_mean(my_n);
                if (my_n > 0) {
                    const char trans = 'T';  // k_qr is column-major
                    const int m_blas = 1;
                    const int n_blas = my_n;
                    const int k_blas = req.m;
                    const double alpha_blas = 1.0;
                    const double beta_blas = 0.0;
                    const int lda = my_n;
                    const int incx = 1;
                    const int incy = 1;

                    // Use dgemv: my_mean = k_qr^T @ alpha
                    dgemv_(&trans, &lda, &k_blas, &alpha_blas, k_qr.data(), &lda,
                           alpha.data(), &incx, &beta_blas, my_mean.data(), &incy);
                }

                // Compute variance: variance - sum(v^2) where v = L^{-1} @ k_qr^T
                // L is lower triangular (m × m), k_qr^T is (m × my_n)
                // Solve L @ V = k_qr^T for V, then var[i] = variance - sum(V[:,i]^2)
                std::vector<double> my_var(my_n);
                if (my_n > 0) {
                    // k_qr is column-major (my_n × m), so k_qr^T is implicitly (m × my_n)
                    // Copy k_qr to a work array since dtrsm modifies in-place
                    std::vector<double> V(req.m * my_n);
                    for (int j = 0; j < my_n; ++j) {
                        for (int i = 0; i < req.m; ++i) {
                            V[i + j * req.m] = k_qr[j + i * my_n];  // Transpose
                        }
                    }

                    // Solve L @ V = V for V (L is lower triangular, m × m)
                    const char side = 'L';
                    const char uplo = 'L';
                    const char transa = 'N';
                    const char diag = 'N';
                    const int m_trsm = req.m;
                    const int n_trsm = my_n;
                    const double alpha_trsm = 1.0;
                    const int lda_trsm = req.m;
                    const int ldb_trsm = req.m;

                    dtrsm_(&side, &uplo, &transa, &diag, &m_trsm, &n_trsm, &alpha_trsm,
                           L_factor.data(), &lda_trsm, V.data(), &ldb_trsm);

                    // Compute variance: variance - ||v||^2 for each column
                    for (int j = 0; j < my_n; ++j) {
                        double sum_sq = 0.0;
                        for (int i = 0; i < req.m; ++i) {
                            double val = V[i + j * req.m];
                            sum_sq += val * val;
                        }
                        my_var[j] = std::max(0.0, req.variance - sum_sq);
                    }
                }

                // Gather results back to rank 0
                std::vector<double> all_mean, all_var;
                if (fit_rank == 0) {
                    all_mean.resize(req.n_query);
                    all_var.resize(req.n_query);
                }

                std::vector<int> recv_counts(fit_size);
                for (int r = 0; r < fit_size; ++r) {
                    int start = r * chunk_size;
                    int end = std::min(start + chunk_size, req.n_query);
                    recv_counts[r] = std::max(0, end - start);
                    displs[r] = start;
                }

                MPI_Gatherv(my_mean.data(), my_n, MPI_DOUBLE,
                            all_mean.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                            0, MPI_COMM_WORLD);
                MPI_Gatherv(my_var.data(), my_n, MPI_DOUBLE,
                            all_var.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                            0, MPI_COMM_WORLD);

                // Write outputs (rank 0 only)
                if (fit_rank == 0) {
                    if (!write_binary_file(req.mean_out_path, all_mean.data(), req.n_query)) {
                        fprintf(stderr, "[Daemon] Failed to write mean\n");
                    }
                    if (!write_binary_file(req.var_out_path, all_var.data(), req.n_query)) {
                        fprintf(stderr, "[Daemon] Failed to write variance\n");
                    }

                    // Send success response
                    DaemonResponse resp;
                    resp.status = 0;
                    resp.message = "Score success";
                    resp.log_marginal_likelihood = 0.0;
                    resp.fit_seconds = 0.0;
                    resp.total_seconds = 0.0;

                    int resp_fd = open(response_pipe, O_WRONLY);
                    if (resp_fd >= 0) {
                        std::string resp_json = create_response_json(resp);
                        write(resp_fd, resp_json.c_str(), resp_json.length());
                        close(resp_fd);
                        fprintf(stderr, "[Daemon] Score response sent\n");
                    }
                }

                MPI_Barrier(MPI_COMM_WORLD);
                continue;
            }

            // Handle fit operation
            // Broadcast operation code (1 = fit)
            int op_code = 1;
            MPI_Bcast(&op_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

            // Broadcast request to all ranks
            MPI_Bcast(&req.m, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&req.d, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&req.length_scale, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&req.variance, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&req.noise, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&req.block_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

            // Load data (rank 0 only)
            std::vector<double> x, y;
            if (!read_binary_file(req.x_path, x, req.m * req.d)) {
                fprintf(stderr, "[Daemon] Failed to read x data\n");
                continue;
            }
            if (!read_binary_file(req.y_path, y, req.m)) {
                fprintf(stderr, "[Daemon] Failed to read y data\n");
                continue;
            }

            fprintf(stderr, "[Daemon] Running fit: m=%d, d=%d, block_size=%d\n",
                    req.m, req.d, req.block_size);

            // Get rank/size for the fit
            int fit_rank, fit_size;
            MPI_Comm_rank(MPI_COMM_WORLD, &fit_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &fit_size);

            // Run fit
#ifdef HAVE_SCALAPACK
            NativeResult result = run_scalapack_distributed(
                static_cast<std::size_t>(req.m),
                fit_rank, fit_size,
                x, y,
                req.d,
                req.length_scale, req.variance, req.noise,
                req.block_size,
                true,   // return_alpha
                true,   // return_chol
                MPI_COMM_WORLD
            );

            double log_marg_lik = result.logdet;
            double fit_time = result.factor_seconds;
            double total_time = result.total_seconds;

            fprintf(stderr, "[Daemon] Fit complete: fit_time=%.4fs, total_time=%.4fs\n",
                    fit_time, total_time);

            // Write outputs (only rank 0 has the results)
            if (!write_binary_file(req.alpha_out_path, result.alpha.data(), req.m)) {
                fprintf(stderr, "[Daemon] Failed to write alpha\n");
            }
            if (!write_binary_file(req.L_out_path, result.chol.data(), req.m * req.m)) {
                fprintf(stderr, "[Daemon] Failed to write L factor\n");
            }

            // Send response
            DaemonResponse resp;
            resp.status = (result.info_potrf == 0 && result.info_potrs == 0) ? 0 : 1;
            resp.message = result.info_potrf == 0 ? "Success" : "Factorization failed";
            resp.log_marginal_likelihood = log_marg_lik;
            resp.fit_seconds = fit_time;
            resp.total_seconds = total_time;

            int resp_fd = open(response_pipe, O_WRONLY);
            if (resp_fd >= 0) {
                std::string resp_json = create_response_json(resp);
                write(resp_fd, resp_json.c_str(), resp_json.length());
                close(resp_fd);
                fprintf(stderr, "[Daemon] Response sent\n");
            } else {
                fprintf(stderr, "[Daemon] Failed to open response pipe: %s\n", strerror(errno));
            }
#else
            fprintf(stderr, "[Daemon] ERROR: Built without ScaLAPACK support\n");
            DaemonResponse resp;
            resp.status = 1;
            resp.message = "Daemon built without ScaLAPACK support";
            resp.log_marginal_likelihood = 0.0;
            resp.fit_seconds = 0.0;
            resp.total_seconds = 0.0;

            int resp_fd = open(response_pipe, O_WRONLY);
            if (resp_fd >= 0) {
                std::string resp_json = create_response_json(resp);
                write(resp_fd, resp_json.c_str(), resp_json.length());
                close(resp_fd);
            }
            continue;
#endif

        } else {
            // Non-root ranks: receive operation code and participate
            int op_code;
            MPI_Bcast(&op_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

            // Handle shutdown op code
            if (op_code == 0) {
                fprintf(stderr, "[Daemon rank %d] Shutdown broadcast received, exiting\n", rank);
                shutdown_requested = 1;
                break;
            }

            if (op_code == 2) {
                // Score operation
                DaemonRequest req;
                MPI_Bcast(&req.n_query, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&req.m, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&req.d, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&req.length_scale, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(&req.variance, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                // Receive broadcasted GP state
                std::vector<double> x_rated(req.m * req.d);
                std::vector<double> alpha(req.m);
                std::vector<double> L_factor(req.m * req.m);

                MPI_Bcast(x_rated.data(), req.m * req.d, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(alpha.data(), req.m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(L_factor.data(), req.m * req.m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                // Receive my chunk of query points
                int fit_rank, fit_size;
                MPI_Comm_rank(MPI_COMM_WORLD, &fit_rank);
                MPI_Comm_size(MPI_COMM_WORLD, &fit_size);

                int chunk_size = (req.n_query + fit_size - 1) / fit_size;
                int my_start = fit_rank * chunk_size;
                int my_end = std::min(my_start + chunk_size, req.n_query);
                int my_n = std::max(0, my_end - my_start);

                std::vector<double> my_x_query(my_n * req.d);
                MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE,
                             my_x_query.data(), my_n * req.d, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                // Compute predictions (same as rank 0)
                std::vector<double> k_qr(my_n * req.m);
                if (my_n > 0) {
                    compute_rbf_kernel(my_x_query.data(), my_n, x_rated.data(), req.m, req.d,
                                      req.length_scale, req.variance, k_qr.data());
                }

                std::vector<double> my_mean(my_n);
                if (my_n > 0) {
                    const char trans = 'T';
                    const int m_blas = 1;
                    const int n_blas = my_n;
                    const int k_blas = req.m;
                    const double alpha_blas = 1.0;
                    const double beta_blas = 0.0;
                    const int lda = my_n;
                    const int incx = 1;
                    const int incy = 1;

                    dgemv_(&trans, &lda, &k_blas, &alpha_blas, k_qr.data(), &lda,
                           alpha.data(), &incx, &beta_blas, my_mean.data(), &incy);
                }

                std::vector<double> my_var(my_n);
                if (my_n > 0) {
                    std::vector<double> V(req.m * my_n);
                    for (int j = 0; j < my_n; ++j) {
                        for (int i = 0; i < req.m; ++i) {
                            V[i + j * req.m] = k_qr[j + i * my_n];
                        }
                    }

                    const char side = 'L';
                    const char uplo = 'L';
                    const char transa = 'N';
                    const char diag = 'N';
                    const int m_trsm = req.m;
                    const int n_trsm = my_n;
                    const double alpha_trsm = 1.0;
                    const int lda_trsm = req.m;
                    const int ldb_trsm = req.m;

                    dtrsm_(&side, &uplo, &transa, &diag, &m_trsm, &n_trsm, &alpha_trsm,
                           L_factor.data(), &lda_trsm, V.data(), &ldb_trsm);

                    for (int j = 0; j < my_n; ++j) {
                        double sum_sq = 0.0;
                        for (int i = 0; i < req.m; ++i) {
                            double val = V[i + j * req.m];
                            sum_sq += val * val;
                        }
                        my_var[j] = std::max(0.0, req.variance - sum_sq);
                    }
                }

                // Gather results
                MPI_Gatherv(my_mean.data(), my_n, MPI_DOUBLE,
                            nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(my_var.data(), my_n, MPI_DOUBLE,
                            nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            } else if (op_code == 1) {
                // Fit operation
                DaemonRequest req;
                MPI_Bcast(&req.m, 1, MPI_INT, 0, MPI_COMM_WORLD);

                if (shutdown_requested) break;

                MPI_Bcast(&req.d, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&req.length_scale, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(&req.variance, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(&req.noise, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(&req.block_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

                int fit_rank, fit_size;
                MPI_Comm_rank(MPI_COMM_WORLD, &fit_rank);
                MPI_Comm_size(MPI_COMM_WORLD, &fit_size);

                std::vector<double> empty_x, empty_y;
#ifdef HAVE_SCALAPACK
                run_scalapack_distributed(
                    static_cast<std::size_t>(req.m),
                    fit_rank, fit_size,
                    empty_x, empty_y,
                    req.d,
                    req.length_scale, req.variance, req.noise,
                    req.block_size,
                    true,   // return_alpha
                    true,   // return_chol
                    MPI_COMM_WORLD
                );
#endif
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        fprintf(stderr, "[Daemon] Shutting down\n");
    }

    MPI_Finalize();
    return 0;
}
