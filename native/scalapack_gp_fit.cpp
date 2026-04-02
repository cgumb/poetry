#include <mpi.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Args {
  std::string input_meta;
  std::string matrix_bin;
  std::string rhs_bin;
  std::string output_meta;
  std::string alpha_bin;
  std::string chol_bin;
};

Args parse_args(int argc, char** argv) {
  std::map<std::string, std::string> kv;
  for (int i = 1; i + 1 < argc; i += 2) {
    kv[argv[i]] = argv[i + 1];
  }
  Args args;
  args.input_meta = kv["--input-meta"];
  args.matrix_bin = kv["--matrix-bin"];
  args.rhs_bin = kv["--rhs-bin"];
  args.output_meta = kv["--output-meta"];
  args.alpha_bin = kv["--alpha-bin"];
  args.chol_bin = kv["--chol-bin"];
  if (args.input_meta.empty() || args.matrix_bin.empty() || args.rhs_bin.empty() ||
      args.output_meta.empty() || args.alpha_bin.empty() || args.chol_bin.empty()) {
    throw std::runtime_error("Missing required arguments for scalapack_gp_fit");
  }
  return args;
}

std::size_t parse_n_from_meta(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Could not open input metadata file");
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  const std::string text = buffer.str();
  const std::string key = "\"n\"";
  auto pos = text.find(key);
  if (pos == std::string::npos) {
    throw std::runtime_error("Input metadata missing field 'n'");
  }
  pos = text.find(':', pos);
  if (pos == std::string::npos) {
    throw std::runtime_error("Could not parse field 'n'");
  }
  ++pos;
  while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
    ++pos;
  }
  std::size_t end = pos;
  while (end < text.size() && std::isdigit(static_cast<unsigned char>(text[end]))) {
    ++end;
  }
  return static_cast<std::size_t>(std::stoull(text.substr(pos, end - pos)));
}

std::vector<double> read_binary_vector(const std::string& path, std::size_t n) {
  std::vector<double> data(n);
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Could not open binary vector file");
  }
  in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(n * sizeof(double)));
  if (!in) {
    throw std::runtime_error("Failed to read expected number of vector bytes");
  }
  return data;
}

std::vector<double> read_binary_matrix(const std::string& path, std::size_t n) {
  std::vector<double> data(n * n);
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Could not open binary matrix file");
  }
  in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(n * n * sizeof(double)));
  if (!in) {
    throw std::runtime_error("Failed to read expected number of matrix bytes");
  }
  return data;
}

void write_binary_vector(const std::string& path, const std::vector<double>& data) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Could not open output vector file");
  }
  out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(double)));
}

void write_binary_matrix(const std::string& path, const std::vector<double>& data) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Could not open output matrix file");
  }
  out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(double)));
}

void build_row_partition(std::size_t n, int size, std::vector<int>& row_counts, std::vector<int>& row_starts) {
  row_counts.assign(size, 0);
  row_starts.assign(size, 0);
  const std::size_t base = n / static_cast<std::size_t>(size);
  const std::size_t rem = n % static_cast<std::size_t>(size);
  int cursor = 0;
  for (int r = 0; r < size; ++r) {
    const std::size_t count = base + (static_cast<std::size_t>(r) < rem ? 1u : 0u);
    row_counts[r] = static_cast<int>(count);
    row_starts[r] = cursor;
    cursor += row_counts[r];
  }
}

int owner_of_row(int global_row, const std::vector<int>& row_counts, const std::vector<int>& row_starts) {
  for (int r = 0; r < static_cast<int>(row_counts.size()); ++r) {
    if (global_row >= row_starts[r] && global_row < row_starts[r] + row_counts[r]) {
      return r;
    }
  }
  return -1;
}

std::vector<double> solve_from_cholesky_lower(const std::vector<double>& l, const std::vector<double>& y, std::size_t n) {
  std::vector<double> z(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    double sum = y[i];
    for (std::size_t k = 0; k < i; ++k) {
      sum -= l[i * n + k] * z[k];
    }
    z[i] = sum / l[i * n + i];
  }

  std::vector<double> x(n, 0.0);
  for (std::size_t ii = 0; ii < n; ++ii) {
    const std::size_t i = n - 1 - ii;
    double sum = z[i];
    for (std::size_t k = i + 1; k < n; ++k) {
      sum -= l[k * n + i] * x[k];
    }
    x[i] = sum / l[i * n + i];
  }
  return x;
}

double logdet_from_cholesky_lower(const std::vector<double>& l, std::size_t n) {
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    sum += std::log(l[i * n + i]);
  }
  return 2.0 * sum;
}

int factorize_lower_mpi_row_partitioned(
    std::vector<double>& local_rows,
    std::size_t n,
    int rank,
    int size,
    const std::vector<int>& row_counts,
    const std::vector<int>& row_starts,
    MPI_Comm comm) {
  int info = 0;
  for (std::size_t j = 0; j < n; ++j) {
    const int global_j = static_cast<int>(j);
    const int owner = owner_of_row(global_j, row_counts, row_starts);
    double ljj = 0.0;
    std::vector<double> row_prefix(j, 0.0);

    if (rank == owner) {
      const int local_j = global_j - row_starts[rank];
      double diag = local_rows[static_cast<std::size_t>(local_j) * n + j];
      for (std::size_t k = 0; k < j; ++k) {
        const double value = local_rows[static_cast<std::size_t>(local_j) * n + k];
        row_prefix[k] = value;
        diag -= value * value;
      }
      if (!(diag > 0.0)) {
        info = global_j + 1;
      } else {
        ljj = std::sqrt(diag);
        local_rows[static_cast<std::size_t>(local_j) * n + j] = ljj;
        for (std::size_t k = j + 1; k < n; ++k) {
          local_rows[static_cast<std::size_t>(local_j) * n + k] = 0.0;
        }
      }
    }

    MPI_Bcast(&info, 1, MPI_INT, owner, comm);
    if (info != 0) {
      break;
    }
    MPI_Bcast(&ljj, 1, MPI_DOUBLE, owner, comm);
    if (j > 0) {
      MPI_Bcast(row_prefix.data(), static_cast<int>(j), MPI_DOUBLE, owner, comm);
    }

    for (int local_i = 0; local_i < row_counts[rank]; ++local_i) {
      const int global_i = row_starts[rank] + local_i;
      if (global_i <= global_j) {
        continue;
      }
      double value = local_rows[static_cast<std::size_t>(local_i) * n + j];
      for (std::size_t k = 0; k < j; ++k) {
        value -= local_rows[static_cast<std::size_t>(local_i) * n + k] * row_prefix[k];
      }
      local_rows[static_cast<std::size_t>(local_i) * n + j] = value / ljj;
    }
  }
  return info;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int exit_code = 0;
  try {
    const Args args = parse_args(argc, argv);
    std::size_t n = 0;
    if (rank == 0) {
      n = parse_n_from_meta(args.input_meta);
    }
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    std::vector<int> row_counts;
    std::vector<int> row_starts;
    build_row_partition(n, size, row_counts, row_starts);

    std::vector<int> sendcounts(size, 0);
    std::vector<int> displs(size, 0);
    for (int r = 0; r < size; ++r) {
      sendcounts[r] = row_counts[r] * static_cast<int>(n);
      displs[r] = row_starts[r] * static_cast<int>(n);
    }

    std::vector<double> full_matrix;
    std::vector<double> rhs;
    if (rank == 0) {
      full_matrix = read_binary_matrix(args.matrix_bin, n);
      rhs = read_binary_vector(args.rhs_bin, n);
    }

    std::vector<double> local_rows(static_cast<std::size_t>(row_counts[rank]) * n, 0.0);
    MPI_Scatterv(
        rank == 0 ? full_matrix.data() : nullptr,
        sendcounts.data(),
        displs.data(),
        MPI_DOUBLE,
        local_rows.data(),
        sendcounts[rank],
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto total_start = std::chrono::steady_clock::now();
    const auto factor_start = std::chrono::steady_clock::now();
    const int info_potrf = factorize_lower_mpi_row_partitioned(
        local_rows,
        n,
        rank,
        size,
        row_counts,
        row_starts,
        MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto factor_end = std::chrono::steady_clock::now();

    std::vector<double> chol;
    if (rank == 0) {
      chol.assign(n * n, 0.0);
    }

    const auto gather_start = std::chrono::steady_clock::now();
    MPI_Gatherv(
        local_rows.data(),
        sendcounts[rank],
        MPI_DOUBLE,
        rank == 0 ? chol.data() : nullptr,
        sendcounts.data(),
        displs.data(),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD);
    const auto gather_end = std::chrono::steady_clock::now();

    if (rank == 0) {
      for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
          chol[i * n + j] = 0.0;
        }
      }

      int info_potrs = -1;
      std::vector<double> alpha(n, 0.0);
      double logdet = 0.0;
      const auto solve_start = std::chrono::steady_clock::now();
      if (info_potrf == 0) {
        alpha = solve_from_cholesky_lower(chol, rhs, n);
        logdet = logdet_from_cholesky_lower(chol, n);
        info_potrs = 0;
      }
      const auto solve_end = std::chrono::steady_clock::now();
      const auto total_end = std::chrono::steady_clock::now();

      write_binary_vector(args.alpha_bin, alpha);
      write_binary_matrix(args.chol_bin, chol);

      const double factor_seconds = std::chrono::duration<double>(factor_end - factor_start).count();
      const double gather_seconds = std::chrono::duration<double>(gather_end - gather_start).count();
      const double solve_seconds = std::chrono::duration<double>(solve_end - solve_start).count();
      const double total_seconds = std::chrono::duration<double>(total_end - total_start).count();

      std::ofstream out(args.output_meta);
      out << "{\n";
      out << "  \"implemented\": true,\n";
      out << "  \"backend\": \"mpi_row_partitioned_reference\",\n";
      out << "  \"message\": \"Multi-rank MPI row-partitioned Cholesky factorization with root gather/solve. BLACS/ScaLAPACK path not implemented yet.\",\n";
      out << "  \"n\": " << n << ",\n";
      out << "  \"world_size\": " << size << ",\n";
      out << "  \"info_potrf\": " << info_potrf << ",\n";
      out << "  \"info_potrs\": " << info_potrs << ",\n";
      out << "  \"factor_seconds\": " << factor_seconds << ",\n";
      out << "  \"solve_seconds\": " << solve_seconds << ",\n";
      out << "  \"gather_seconds\": " << gather_seconds << ",\n";
      out << "  \"total_seconds\": " << total_seconds << ",\n";
      out << "  \"logdet\": " << logdet << "\n";
      out << "}\n";
      std::cerr << "[scalapack_gp_fit mpi] Completed multi-rank row-partitioned factorization for n=" << n << " using " << size << " MPI ranks.\n";
      if (info_potrf != 0) {
        exit_code = 2;
      }
    }
  } catch (const std::exception& ex) {
    if (rank == 0) {
      std::cerr << "scalapack_gp_fit failed: " << ex.what() << "\n";
    }
    exit_code = 1;
  }

  MPI_Finalize();
  return exit_code;
}
