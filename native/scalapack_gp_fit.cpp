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

int cholesky_lower_inplace(std::vector<double>& a, std::size_t n) {
  for (std::size_t j = 0; j < n; ++j) {
    double sum = a[j * n + j];
    for (std::size_t k = 0; k < j; ++k) {
      const double ljk = a[j * n + k];
      sum -= ljk * ljk;
    }
    if (!(sum > 0.0)) {
      return static_cast<int>(j + 1);
    }
    const double ljj = std::sqrt(sum);
    a[j * n + j] = ljj;
    for (std::size_t i = j + 1; i < n; ++i) {
      double s = a[i * n + j];
      for (std::size_t k = 0; k < j; ++k) {
        s -= a[i * n + k] * a[j * n + k];
      }
      a[i * n + j] = s / ljj;
    }
    for (std::size_t k = j + 1; k < n; ++k) {
      a[j * n + k] = 0.0;
    }
  }
  return 0;
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

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int exit_code = 0;
  try {
    const Args args = parse_args(argc, argv);
    if (rank == 0) {
      const std::size_t n = parse_n_from_meta(args.input_meta);
      const auto matrix = read_binary_matrix(args.matrix_bin, n);
      const auto rhs = read_binary_vector(args.rhs_bin, n);

      auto chol = matrix;
      const auto t0 = std::chrono::steady_clock::now();
      const auto factor_start = std::chrono::steady_clock::now();
      const int info_potrf = cholesky_lower_inplace(chol, n);
      const auto factor_end = std::chrono::steady_clock::now();

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
      const auto gather_start = std::chrono::steady_clock::now();
      write_binary_vector(args.alpha_bin, alpha);
      write_binary_matrix(args.chol_bin, chol);
      const auto gather_end = std::chrono::steady_clock::now();
      const auto t1 = std::chrono::steady_clock::now();

      const double factor_seconds = std::chrono::duration<double>(factor_end - factor_start).count();
      const double solve_seconds = std::chrono::duration<double>(solve_end - solve_start).count();
      const double gather_seconds = std::chrono::duration<double>(gather_end - gather_start).count();
      const double total_seconds = std::chrono::duration<double>(t1 - t0).count();

      std::ofstream out(args.output_meta);
      out << "{\n";
      out << "  \"implemented\": true,\n";
      out << "  \"backend\": \"native_serial_reference\",\n";
      out << "  \"message\": \"Numerical reference path on rank 0 only; distributed ScaLAPACK path not implemented yet.\",\n";
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
      std::cerr << "[scalapack_gp_fit reference] Completed rank-0 reference solve for n=" << n << " using " << size << " MPI ranks.\n";
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
