#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef HAVE_SCALAPACK
extern "C" {
void Cblacs_pinfo(int* mypnum, int* nprocs);
void Cblacs_get(int context, int request, int* value);
void Cblacs_gridinit(int* context, const char* order, int nprow, int npcol);
void Cblacs_gridinfo(int context, int* nprow, int* npcol, int* myrow, int* mycol);
void Cblacs_gridexit(int context);
int numroc_(const int* n, const int* nb, const int* iproc, const int* isrcproc, const int* nprocs);
void descinit_(int* desc, const int* m, const int* n, const int* mb, const int* nb, const int* irsrc, const int* icsrc, const int* ictxt, const int* lld, int* info);
void pdpotrf_(const char* uplo, const int* n, double* a, const int* ia, const int* ja, const int* desca, int* info);
void pdpotrs_(const char* uplo, const int* n, const int* nrhs, const double* a, const int* ia, const int* ja, const int* desca, double* b, const int* ib, const int* jb, const int* descb, int* info);
}
#endif

struct Args {
  std::string input_meta;
  std::string matrix_bin;
  std::string rhs_bin;
  std::string output_meta;
  std::string alpha_bin;
  std::string chol_bin;
  std::string backend = "auto";
  int block_size = 128;
};

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
  if (kv.count("--backend")) {
    args.backend = kv["--backend"];
  }
  if (kv.count("--block-size")) {
    args.block_size = std::stoi(kv["--block-size"]);
  }
  if (args.input_meta.empty() || args.matrix_bin.empty() || args.rhs_bin.empty() ||
      args.output_meta.empty() || args.alpha_bin.empty() || args.chol_bin.empty()) {
    throw std::runtime_error("Missing required arguments for scalapack_gp_fit");
  }
  return args;
}

bool compiled_with_scalapack() {
#ifdef HAVE_SCALAPACK
  return true;
#else
  return false;
#endif
}

std::string normalize_backend_name(const std::string& backend) {
  if (backend == "auto") {
    return compiled_with_scalapack() ? "scalapack" : "mpi_row_partitioned_reference";
  }
  if (backend == "mpi" || backend == "native_reference") {
    return "mpi_row_partitioned_reference";
  }
  if (backend == "mpi_row_partitioned_reference" || backend == "scalapack") {
    return backend;
  }
  throw std::runtime_error("Unknown backend requested: " + backend);
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

void write_output_meta(const std::string& path, const NativeResult& result, std::size_t n, int size) {
  std::ofstream out(path);
  out << "{\n";
  out << "  \"implemented\": " << (result.implemented ? "true" : "false") << ",\n";
  out << "  \"backend\": \"" << result.backend << "\",\n";
  out << "  \"requested_backend\": \"" << result.requested_backend << "\",\n";
  out << "  \"compiled_with_scalapack\": " << (compiled_with_scalapack() ? "true" : "false") << ",\n";
  out << "  \"message\": \"" << result.message << "\",\n";
  out << "  \"n\": " << n << ",\n";
  out << "  \"world_size\": " << size << ",\n";
  out << "  \"info_potrf\": " << result.info_potrf << ",\n";
  out << "  \"info_potrs\": " << result.info_potrs << ",\n";
  out << "  \"factor_seconds\": " << result.factor_seconds << ",\n";
  out << "  \"solve_seconds\": " << result.solve_seconds << ",\n";
  out << "  \"gather_seconds\": " << result.gather_seconds << ",\n";
  out << "  \"total_seconds\": " << result.total_seconds << ",\n";
  out << "  \"logdet\": " << result.logdet << "\n";
  out << "}\n";
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

int factorize_lower_mpi_row_partitioned(std::vector<double>& local_rows, std::size_t n, int rank, int size, const std::vector<int>& row_counts, const std::vector<int>& row_starts, MPI_Comm comm) {
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

NativeResult run_mpi_reference(std::size_t n, int rank, int size, const std::vector<double>& full_matrix_root, const std::vector<double>& rhs_root, MPI_Comm comm) {
  NativeResult result;
  result.implemented = true;
  result.backend = "mpi_row_partitioned_reference";
  result.message = "Multi-rank MPI row-partitioned Cholesky factorization with root gather/solve. BLACS/ScaLAPACK path not implemented yet.";

  std::vector<int> row_counts;
  std::vector<int> row_starts;
  build_row_partition(n, size, row_counts, row_starts);

  std::vector<int> sendcounts(size, 0);
  std::vector<int> displs(size, 0);
  for (int r = 0; r < size; ++r) {
    sendcounts[r] = row_counts[r] * static_cast<int>(n);
    displs[r] = row_starts[r] * static_cast<int>(n);
  }

  std::vector<double> local_rows(static_cast<std::size_t>(row_counts[rank]) * n, 0.0);
  MPI_Scatterv(rank == 0 ? full_matrix_root.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE, local_rows.data(), sendcounts[rank], MPI_DOUBLE, 0, comm);

  MPI_Barrier(comm);
  const auto total_start = std::chrono::steady_clock::now();
  const auto factor_start = std::chrono::steady_clock::now();
  result.info_potrf = factorize_lower_mpi_row_partitioned(local_rows, n, rank, size, row_counts, row_starts, comm);
  MPI_Barrier(comm);
  const auto factor_end = std::chrono::steady_clock::now();

  std::vector<double> chol;
  if (rank == 0) {
    chol.assign(n * n, 0.0);
  }

  const auto gather_start = std::chrono::steady_clock::now();
  MPI_Gatherv(local_rows.data(), sendcounts[rank], MPI_DOUBLE, rank == 0 ? chol.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE, 0, comm);
  const auto gather_end = std::chrono::steady_clock::now();

  if (rank == 0) {
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i + 1; j < n; ++j) {
        chol[i * n + j] = 0.0;
      }
    }

    const auto solve_start = std::chrono::steady_clock::now();
    if (result.info_potrf == 0) {
      result.alpha = solve_from_cholesky_lower(chol, rhs_root, n);
      result.logdet = logdet_from_cholesky_lower(chol, n);
      result.info_potrs = 0;
    }
    const auto solve_end = std::chrono::steady_clock::now();
    const auto total_end = std::chrono::steady_clock::now();
    result.chol = std::move(chol);
    result.factor_seconds = std::chrono::duration<double>(factor_end - factor_start).count();
    result.gather_seconds = std::chrono::duration<double>(gather_end - gather_start).count();
    result.solve_seconds = std::chrono::duration<double>(solve_end - solve_start).count();
    result.total_seconds = std::chrono::duration<double>(total_end - total_start).count();
  }
  return result;
}

#ifdef HAVE_SCALAPACK
int choose_nprow(int size) {
  int nprow = static_cast<int>(std::floor(std::sqrt(static_cast<double>(size))));
  while (nprow > 1 && size % nprow != 0) {
    --nprow;
  }
  return std::max(1, nprow);
}

int block_owner(int gidx, int nb, int nprocs_dim) {
  return (gidx / nb) % nprocs_dim;
}

int local_index(int gidx, int nb, int nprocs_dim) {
  const int block = gidx / nb;
  const int inblock = gidx % nb;
  const int local_block = block / nprocs_dim;
  return local_block * nb + inblock;
}

NativeResult run_scalapack(std::size_t n, int rank, int size, const std::vector<double>& full_matrix_root, const std::vector<double>& rhs_root, int block_size, MPI_Comm comm) {
  NativeResult result;
  result.backend = "scalapack";
  result.implemented = true;
  result.message = "Distributed ScaLAPACK Cholesky factorization and solve with root-side reconstruction of outputs.";

  const int n_int = static_cast<int>(n);
  const int nb = std::max(1, block_size);
  const int nprow = choose_nprow(size);
  const int npcol = size / nprow;
  int ictxt = 0;
  Cblacs_get(-1, 0, &ictxt);
  Cblacs_gridinit(&ictxt, "R", nprow, npcol);
  int myrow = -1;
  int mycol = -1;
  int grid_rows = 0;
  int grid_cols = 0;
  Cblacs_gridinfo(ictxt, &grid_rows, &grid_cols, &myrow, &mycol);

  const int rsrc = 0;
  const int csrc = 0;
  const int nrhs = 1;
  const int local_rows = numroc_(&n_int, &nb, &myrow, &rsrc, &grid_rows);
  const int local_cols = numroc_(&n_int, &nb, &mycol, &csrc, &grid_cols);
  const int local_rhs_cols = numroc_(&nrhs, &nb, &mycol, &csrc, &grid_cols);
  const int lld_a = std::max(1, local_rows);
  const int lld_b = std::max(1, local_rows);

  std::vector<double> local_a(static_cast<std::size_t>(lld_a) * std::max(1, local_cols), 0.0);
  std::vector<double> local_b(static_cast<std::size_t>(lld_b) * std::max(1, local_rhs_cols), 0.0);

  int desc_a[9];
  int desc_b[9];
  int info_desc_a = 0;
  int info_desc_b = 0;
  descinit_(desc_a, &n_int, &n_int, &nb, &nb, &rsrc, &csrc, &ictxt, &lld_a, &info_desc_a);
  descinit_(desc_b, &n_int, &nrhs, &nb, &nb, &rsrc, &csrc, &ictxt, &lld_b, &info_desc_b);
  if (info_desc_a != 0 || info_desc_b != 0) {
    result.implemented = false;
    result.message = "ScaLAPACK descriptor initialization failed.";
    Cblacs_gridexit(ictxt);
    return result;
  }

  std::vector<double> full_matrix;
  std::vector<double> rhs;
  if (rank == 0) {
    full_matrix = full_matrix_root;
    rhs = rhs_root;
  } else {
    full_matrix.assign(n * n, 0.0);
    rhs.assign(n, 0.0);
  }
  MPI_Bcast(full_matrix.data(), static_cast<int>(n * n), MPI_DOUBLE, 0, comm);
  MPI_Bcast(rhs.data(), static_cast<int>(n), MPI_DOUBLE, 0, comm);

  for (int gi = 0; gi < n_int; ++gi) {
    for (int gj = 0; gj < n_int; ++gj) {
      if (block_owner(gi, nb, grid_rows) == myrow && block_owner(gj, nb, grid_cols) == mycol) {
        const int li = local_index(gi, nb, grid_rows);
        const int lj = local_index(gj, nb, grid_cols);
        local_a[static_cast<std::size_t>(li) + static_cast<std::size_t>(lj) * lld_a] = full_matrix[static_cast<std::size_t>(gi) * n + gj];
      }
    }
    if (block_owner(gi, nb, grid_rows) == myrow && block_owner(0, nb, grid_cols) == mycol) {
      const int li = local_index(gi, nb, grid_rows);
      const int lj = local_index(0, nb, grid_cols);
      local_b[static_cast<std::size_t>(li) + static_cast<std::size_t>(lj) * lld_b] = rhs[gi];
    }
  }

  MPI_Barrier(comm);
  const auto total_start = std::chrono::steady_clock::now();
  const auto factor_start = std::chrono::steady_clock::now();
  const char uplo = 'L';
  const int ia = 1;
  const int ja = 1;
  pdpotrf_(&uplo, &n_int, local_a.data(), &ia, &ja, desc_a, &result.info_potrf);
  MPI_Barrier(comm);
  const auto factor_end = std::chrono::steady_clock::now();

  const auto solve_start = std::chrono::steady_clock::now();
  if (result.info_potrf == 0) {
    const int ib = 1;
    const int jb = 1;
    pdpotrs_(&uplo, &n_int, &nrhs, local_a.data(), &ia, &ja, desc_a, local_b.data(), &ib, &jb, desc_b, &result.info_potrs);
  }
  MPI_Barrier(comm);
  const auto solve_end = std::chrono::steady_clock::now();

  const auto gather_start = std::chrono::steady_clock::now();
  std::vector<double> partial_chol(n * n, 0.0);
  std::vector<double> partial_alpha(n, 0.0);
  for (int gi = 0; gi < n_int; ++gi) {
    for (int gj = 0; gj < n_int; ++gj) {
      if (block_owner(gi, nb, grid_rows) == myrow && block_owner(gj, nb, grid_cols) == mycol) {
        const int li = local_index(gi, nb, grid_rows);
        const int lj = local_index(gj, nb, grid_cols);
        partial_chol[static_cast<std::size_t>(gi) * n + gj] = local_a[static_cast<std::size_t>(li) + static_cast<std::size_t>(lj) * lld_a];
      }
    }
    if (result.info_potrf == 0 && block_owner(gi, nb, grid_rows) == myrow && block_owner(0, nb, grid_cols) == mycol) {
      const int li = local_index(gi, nb, grid_rows);
      const int lj = local_index(0, nb, grid_cols);
      partial_alpha[gi] = local_b[static_cast<std::size_t>(li) + static_cast<std::size_t>(lj) * lld_b];
    }
  }

  std::vector<double> gathered_chol;
  std::vector<double> gathered_alpha;
  if (rank == 0) {
    gathered_chol.assign(n * n, 0.0);
    gathered_alpha.assign(n, 0.0);
  }
  MPI_Reduce(partial_chol.data(), rank == 0 ? gathered_chol.data() : nullptr, static_cast<int>(n * n), MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(partial_alpha.data(), rank == 0 ? gathered_alpha.data() : nullptr, static_cast<int>(n), MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Barrier(comm);
  const auto gather_end = std::chrono::steady_clock::now();
  const auto total_end = std::chrono::steady_clock::now();

  Cblacs_gridexit(ictxt);

  if (rank == 0) {
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i + 1; j < n; ++j) {
        gathered_chol[i * n + j] = 0.0;
      }
    }
    result.alpha = std::move(gathered_alpha);
    result.chol = std::move(gathered_chol);
    if (result.info_potrf == 0 && result.info_potrs == 0) {
      result.logdet = logdet_from_cholesky_lower(result.chol, n);
    }
    result.factor_seconds = std::chrono::duration<double>(factor_end - factor_start).count();
    result.solve_seconds = std::chrono::duration<double>(solve_end - solve_start).count();
    result.gather_seconds = std::chrono::duration<double>(gather_end - gather_start).count();
    result.total_seconds = std::chrono::duration<double>(total_end - total_start).count();
  }
  return result;
}
#endif

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
    std::size_t n = 0;
    if (rank == 0) {
      n = parse_n_from_meta(args.input_meta);
    }
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    std::vector<double> full_matrix;
    std::vector<double> rhs;
    if (rank == 0) {
      full_matrix = read_binary_matrix(args.matrix_bin, n);
      rhs = read_binary_vector(args.rhs_bin, n);
    }

    NativeResult result;
    if (resolved_backend == "scalapack") {
#ifdef HAVE_SCALAPACK
      result = run_scalapack(n, rank, size, full_matrix, rhs, args.block_size, MPI_COMM_WORLD);
#else
      result.implemented = false;
      result.backend = "scalapack";
      result.message = "ScaLAPACK backend requested, but the executable was not built with ScaLAPACK support.";
#endif
    } else {
      result = run_mpi_reference(n, rank, size, full_matrix, rhs, MPI_COMM_WORLD);
    }
    result.requested_backend = args.backend;

    if (rank == 0) {
      if (result.alpha.empty()) {
        result.alpha.assign(n, 0.0);
      }
      if (result.chol.empty()) {
        result.chol.assign(n * n, 0.0);
      }
      write_binary_vector(args.alpha_bin, result.alpha);
      write_binary_matrix(args.chol_bin, result.chol);
      write_output_meta(args.output_meta, result, n, size);
      std::cerr << "[scalapack_gp_fit] backend=" << result.backend << " requested=" << result.requested_backend << " n=" << n << " ranks=" << size << "\n";
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
