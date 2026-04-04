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

// Forward declarations from scalapack_gp_fit.cpp
extern "C" {
void run_scalapack_distributed(
    const double* x,
    const double* y,
    int m,
    int d,
    double length_scale,
    double variance,
    double noise,
    int nprow,
    int npcol,
    int block_size,
    double* alpha_out,
    double* L_factor_out,
    double* log_marginal_likelihood_out,
    double* timing_out
);
}

volatile sig_atomic_t shutdown_requested = 0;

void signal_handler(int signum) {
    shutdown_requested = 1;
}

struct DaemonRequest {
    std::string operation;  // "fit", "shutdown"
    int m;
    int d;
    double length_scale;
    double variance;
    double noise;
    int nprow;
    int npcol;
    int block_size;
    std::string x_path;
    std::string y_path;
    std::string alpha_out_path;
    std::string L_out_path;
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
    // Very simple parser - expects exact format
    // Real implementation would use a proper JSON library

    // Find operation
    size_t op_start = json.find("\"operation\":\"") + 13;
    size_t op_end = json.find("\"", op_start);
    if (op_start == std::string::npos || op_end == std::string::npos) return false;
    req.operation = json.substr(op_start, op_end - op_start);

    if (req.operation == "shutdown") {
        return true;
    }

    // Parse integers
    auto parse_int = [&](const char* key, int& val) {
        std::string key_str = std::string("\"") + key + "\":";
        size_t pos = json.find(key_str);
        if (pos == std::string::npos) return false;
        pos += key_str.length();
        val = std::atoi(json.c_str() + pos);
        return true;
    };

    // Parse doubles
    auto parse_double = [&](const char* key, double& val) {
        std::string key_str = std::string("\"") + key + "\":";
        size_t pos = json.find(key_str);
        if (pos == std::string::npos) return false;
        pos += key_str.length();
        val = std::atof(json.c_str() + pos);
        return true;
    };

    // Parse strings
    auto parse_string = [&](const char* key, std::string& val) {
        std::string key_str = std::string("\"") + key + "\":\"";
        size_t start = json.find(key_str);
        if (start == std::string::npos) return false;
        start += key_str.length();
        size_t end = json.find("\"", start);
        if (end == std::string::npos) return false;
        val = json.substr(start, end - start);
        return true;
    };

    if (!parse_int("m", req.m)) return false;
    if (!parse_int("d", req.d)) return false;
    if (!parse_double("length_scale", req.length_scale)) return false;
    if (!parse_double("variance", req.variance)) return false;
    if (!parse_double("noise", req.noise)) return false;
    if (!parse_int("nprow", req.nprow)) return false;
    if (!parse_int("npcol", req.npcol)) return false;
    if (!parse_int("block_size", req.block_size)) return false;
    if (!parse_string("x_path", req.x_path)) return false;
    if (!parse_string("y_path", req.y_path)) return false;
    if (!parse_string("alpha_out_path", req.alpha_out_path)) return false;
    if (!parse_string("L_out_path", req.L_out_path)) return false;

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

            // Broadcast request to all ranks
            MPI_Bcast(&req.m, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&req.d, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&req.length_scale, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&req.variance, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&req.noise, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&req.nprow, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&req.npcol, 1, MPI_INT, 0, MPI_COMM_WORLD);
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

            // Prepare output buffers
            std::vector<double> alpha(req.m);
            std::vector<double> L_factor(req.m * req.m);
            double log_marg_lik = 0.0;
            double timing[4] = {0.0};

            fprintf(stderr, "[Daemon] Running fit: m=%d, d=%d, nprow=%d, npcol=%d, block_size=%d\n",
                    req.m, req.d, req.nprow, req.npcol, req.block_size);

            // Run fit
            run_scalapack_distributed(
                x.data(), y.data(),
                req.m, req.d,
                req.length_scale, req.variance, req.noise,
                req.nprow, req.npcol, req.block_size,
                alpha.data(), L_factor.data(),
                &log_marg_lik, timing
            );

            fprintf(stderr, "[Daemon] Fit complete: fit_time=%.4fs, total_time=%.4fs\n",
                    timing[0], timing[3]);

            // Write outputs
            if (!write_binary_file(req.alpha_out_path, alpha.data(), req.m)) {
                fprintf(stderr, "[Daemon] Failed to write alpha\n");
            }
            if (!write_binary_file(req.L_out_path, L_factor.data(), req.m * req.m)) {
                fprintf(stderr, "[Daemon] Failed to write L factor\n");
            }

            // Send response
            DaemonResponse resp;
            resp.status = 0;
            resp.message = "Success";
            resp.log_marginal_likelihood = log_marg_lik;
            resp.fit_seconds = timing[0];
            resp.total_seconds = timing[3];

            int resp_fd = open(response_pipe, O_WRONLY);
            if (resp_fd >= 0) {
                std::string resp_json = create_response_json(resp);
                write(resp_fd, resp_json.c_str(), resp_json.length());
                close(resp_fd);
                fprintf(stderr, "[Daemon] Response sent\n");
            } else {
                fprintf(stderr, "[Daemon] Failed to open response pipe: %s\n", strerror(errno));
            }

        } else {
            // Non-root ranks: receive broadcast and participate in computation
            DaemonRequest req;
            MPI_Bcast(&req.m, 1, MPI_INT, 0, MPI_COMM_WORLD);

            if (shutdown_requested) break;

            MPI_Bcast(&req.d, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&req.length_scale, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&req.variance, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&req.noise, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&req.nprow, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&req.npcol, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&req.block_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

            // Participate in computation (data will be broadcast inside run_scalapack_distributed)
            run_scalapack_distributed(
                nullptr, nullptr,  // Data broadcast inside
                req.m, req.d,
                req.length_scale, req.variance, req.noise,
                req.nprow, req.npcol, req.block_size,
                nullptr, nullptr, nullptr, nullptr  // Only root needs outputs
            );
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        fprintf(stderr, "[Daemon] Shutting down\n");
    }

    MPI_Finalize();
    return 0;
}
