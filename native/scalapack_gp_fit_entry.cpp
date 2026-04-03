#define main scalapack_gp_fit_legacy_main
#include "scalapack_gp_fit.cpp"
#undef main

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

// BLAS dgemm for optimized kernel assembly
extern "C" {
  void dgemm_(const char* transa, const char* transb,
              const int* m, const int* n, const int* k,
              const double* alpha, const double* a, const int* lda,
              const double* b, const int* ldb,
              const double* beta, double* c, const int* ldc);
}

namespace {

struct EntryInputMeta {
  std::size_t n = 0;
  int d = 0;
  std::string input_kind = "matrix";
  double length_scale = 1.0;
  double variance = 1.0;
  double noise = 1e-3;
};

std::string read_text_file_entry(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Could not open input metadata file");
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

std::string parse_string_field_entry(const std::string& text, const std::string& key, const std::string& fallback) {
  const std::string quoted = "\"" + key + "\"";
  auto key_pos = text.find(quoted);
  if (key_pos == std::string::npos) {
    return fallback;
  }
  auto pos = text.find(':', key_pos);
  if (pos == std::string::npos) {
    return fallback;
  }
  ++pos;
  while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
    ++pos;
  }
  if (pos >= text.size() || text[pos] != '"') {
    return fallback;
  }
  ++pos;
  auto end = text.find('"', pos);
  if (end == std::string::npos) {
    return fallback;
  }
  return text.substr(pos, end - pos);
}

int parse_int_field_entry(const std::string& text, const std::string& key, int fallback) {
  const std::string quoted = "\"" + key + "\"";
  auto key_pos = text.find(quoted);
  if (key_pos == std::string::npos) {
    return fallback;
  }
  auto pos = text.find(':', key_pos);
  if (pos == std::string::npos) {
    return fallback;
  }
  ++pos;
  while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
    ++pos;
  }
  std::size_t end = pos;
  if (end < text.size() && (text[end] == '-' || text[end] == '+')) {
    ++end;
  }
  while (end < text.size() && std::isdigit(static_cast<unsigned char>(text[end]))) {
    ++end;
  }
  if (end == pos) {
    return fallback;
  }
  return std::stoi(text.substr(pos, end - pos));
}

double parse_double_field_entry(const std::string& text, const std::string& key, double fallback) {
  const std::string quoted = "\"" + key + "\"";
  auto key_pos = text.find(quoted);
  if (key_pos == std::string::npos) {
    return fallback;
  }
  auto pos = text.find(':', key_pos);
  if (pos == std::string::npos) {
    return fallback;
  }
  ++pos;
  while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
    ++pos;
  }
  std::size_t end = pos;
  while (end < text.size() && (std::isdigit(static_cast<unsigned char>(text[end])) || text[end] == '-' || text[end] == '+' || text[end] == '.' || text[end] == 'e' || text[end] == 'E')) {
    ++end;
  }
  if (end == pos) {
    return fallback;
  }
  return std::stod(text.substr(pos, end - pos));
}

EntryInputMeta parse_entry_input_meta(const std::string& path) {
  const std::string text = read_text_file_entry(path);
  EntryInputMeta meta;
  meta.n = parse_n_from_meta(path);
  meta.input_kind = parse_string_field_entry(text, "input_kind", "matrix");
  meta.d = parse_int_field_entry(text, "d", 0);
  meta.length_scale = parse_double_field_entry(text, "length_scale", 1.0);
  meta.variance = parse_double_field_entry(text, "variance", 1.0);
  meta.noise = parse_double_field_entry(text, "noise", 1e-3);
  return meta;
}

std::vector<double> read_binary_feature_matrix_entry(const std::string& path, std::size_t n, int d) {
  std::vector<double> data(n * static_cast<std::size_t>(d));
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Could not open binary feature matrix file");
  }
  in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(double)));
  if (!in) {
    throw std::runtime_error("Failed to read expected number of feature bytes");
  }
  return data;
}

std::vector<double> build_dense_rbf_matrix_from_features_entry(
    const std::vector<double>& x,
    std::size_t n,
    int d,
    double length_scale,
    double variance,
    double noise) {
  std::cerr << "[DEBUG] build_dense_rbf_matrix: n=" << n << " d=" << d << " x.size()=" << x.size() << std::endl;

  // Verify x has correct size
  if (x.size() != n * static_cast<std::size_t>(d)) {
    std::cerr << "[ERROR] x.size()=" << x.size() << " but expected n*d=" << (n * d) << std::endl;
    throw std::runtime_error("Feature matrix size mismatch");
  }

  // Step 1: Compute row norms ||x[i]||^2
  std::cerr << "[DEBUG] Computing norms..." << std::endl;
  std::vector<double> norms(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    double sum = 0.0;
    for (int k = 0; k < d; ++k) {
      const double value = x[i * static_cast<std::size_t>(d) + static_cast<std::size_t>(k)];
      sum += value * value;
    }
    norms[i] = sum;
  }
  std::cerr << "[DEBUG] Norms computed" << std::endl;

  // Step 2: Compute Gram matrix G = X @ X^T
  // TODO: Optimize this with BLAS DGEMM after fixing row/column-major issues
  // For now, use simple but correct loop
  std::cerr << "[DEBUG] Computing Gram matrix (simple loop)..." << std::endl;
  std::vector<double> gram(n * n, 0.0);

  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double dot = 0.0;
      for (int k = 0; k < d; ++k) {
        dot += x[i * static_cast<std::size_t>(d) + static_cast<std::size_t>(k)] *
               x[j * static_cast<std::size_t>(d) + static_cast<std::size_t>(k)];
      }
      gram[i * n + j] = dot;
    }
  }

  std::cerr << "[DEBUG] Gram matrix computed" << std::endl;

  // Step 3: Apply RBF kernel transformation
  // K[i,j] = variance * exp(-0.5 * ||x[i] - x[j]||^2 / length_scale^2)
  // Using: ||x[i] - x[j]||^2 = ||x[i]||^2 + ||x[j]||^2 - 2 * <x[i], x[j]>
  std::vector<double> matrix(n * n, 0.0);
  const double inv_two_ell_sq = -0.5 / (length_scale * length_scale);

  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double d2 = norms[i] + norms[j] - 2.0 * gram[i * n + j];
      if (d2 < 0.0) {
        d2 = 0.0;  // Numerical stability for near-identical points
      }
      matrix[i * n + j] = variance * std::exp(inv_two_ell_sq * d2);
    }
  }

  // Step 4: Add noise to diagonal
  std::cerr << "[DEBUG] Adding noise to diagonal..." << std::endl;
  for (std::size_t i = 0; i < n; ++i) {
    matrix[i * n + i] += noise * noise;
  }

  std::cerr << "[DEBUG] build_dense_rbf_matrix completed successfully" << std::endl;
  return matrix;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int exit_code = 0;
  try {
    const Args args = parse_args(argc, argv);
    const std::string resolved_backend = normalize_backend_name(args.backend);

    EntryInputMeta meta;
    if (rank == 0) {
      meta = parse_entry_input_meta(args.input_meta);
    }

    unsigned long long n64 = static_cast<unsigned long long>(meta.n);
    int d = meta.d;
    double length_scale = meta.length_scale;
    double variance = meta.variance;
    double noise = meta.noise;
    int input_kind_code = (meta.input_kind == "features") ? 1 : 0;

    MPI_Bcast(&n64, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&length_scale, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&variance, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&noise, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&input_kind_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

    meta.n = static_cast<std::size_t>(n64);
    meta.d = d;
    meta.length_scale = length_scale;
    meta.variance = variance;
    meta.noise = noise;
    meta.input_kind = input_kind_code == 1 ? "features" : "matrix";

    std::vector<double> full_matrix;
    std::vector<double> rhs;
    if (rank == 0) {
      rhs = read_binary_vector(args.rhs_bin, meta.n);
      if (meta.input_kind == "features") {
        const std::vector<double> x_rated = read_binary_feature_matrix_entry(args.matrix_bin, meta.n, meta.d);
        full_matrix = build_dense_rbf_matrix_from_features_entry(
            x_rated, meta.n, meta.d, meta.length_scale, meta.variance, meta.noise);
      } else {
        full_matrix = read_binary_matrix(args.matrix_bin, meta.n);
      }
    }

    NativeResult result;
    if (resolved_backend == "scalapack") {
#ifdef HAVE_SCALAPACK
      result = run_scalapack(meta.n, rank, size, full_matrix, rhs, args.block_size, MPI_COMM_WORLD);
#else
      result.implemented = false;
      result.backend = "scalapack";
      result.message = "ScaLAPACK backend requested, but the executable was not built with ScaLAPACK support.";
#endif
    } else {
      result = run_mpi_reference(meta.n, rank, size, full_matrix, rhs, MPI_COMM_WORLD);
    }
    result.requested_backend = args.backend;

    if (rank == 0) {
      if (result.alpha.empty()) {
        result.alpha.assign(meta.n, 0.0);
      }
      if (result.chol.empty()) {
        result.chol.assign(meta.n * meta.n, 0.0);
      }
      write_binary_vector(args.alpha_bin, result.alpha);
      write_binary_matrix(args.chol_bin, result.chol);
      write_output_meta(args.output_meta, result, meta.n, size);
      std::cerr << "[scalapack_gp_fit] backend=" << result.backend
                << " requested=" << result.requested_backend
                << " n=" << meta.n
                << " ranks=" << size << "\n";
      if (!result.implemented) {
        exit_code = 3;
      } else if (result.info_potrf != 0 || result.info_potrs != 0) {
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
