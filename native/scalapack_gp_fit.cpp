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
namespace {
constexpr int TAG_MATRIX_HEADER = 100;
constexpr int TAG_MATRIX_TILE = 101;
constexpr int TAG_RHS_VECTOR = 110;
constexpr int TAG_GATHER_MATRIX = 200;
constexpr int TAG_GATHER_ALPHA = 210;
}

int numroc_wrapper(int n, int nb, int iproc, int isrcproc, int nprocs) {
  return numroc_(&n, &nb, &iproc, &isrcproc, &nprocs);
}

int choose_nprow(int size) {
  int nprow = static_cast<int>(std::floor(std::sqrt(static_cast<double>(size))));
  while (nprow > 1 && size % nprow != 0) {
    --nprow;
  }
  return std::max(1, nprow);
}

int grid_rank(int prow, int pcol, int npcol) {
  return prow * npcol + pcol;
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

int owned_tile_count(int n, int nb, int myrow, int mycol, int nprow, int npcol) {
  int count = 0;
  for (int gi0 = myrow * nb; gi0 < n; gi0 += nprow * nb) {
    for (int gj0 = mycol * nb; gj0 < n; gj0 += npcol * nb) {
      ++count;
    }
  }
  return count;
}

void copy_global_tile_to_local_column_major(
    const std::vector<double>& global_matrix,
    std::size_t n,
    int global_i0,
    int global_j0,
    int rows,
    int cols,
    std::vector<double>& local_matrix,
    int lld_local,
    int local_i0,
    int local_j0) {
  for (int dj = 0; dj < cols; ++dj) {
    for (int di = 0; di < rows; ++di) {
      local_matrix[static_cast<std::size_t>(local_i0 + di) + static_cast<std::size_t>(local_j0 + dj) * lld_local] =
          global_matrix[static_cast<std::size_t>(global_i0 + di) * n + static_cast<std::size_t>(global_j0 + dj)];
    }
  }
}

void pack_column_major_tile_from_global(
    const std::vector<double>& global_matrix,
    std::size_t n,
    int global_i0,
    int global_j0,
    int rows,
    int cols,
    std::vector<double>& tile) {
  tile.assign(static_cast<std::size_t>(rows) * cols, 0.0);
  for (int dj = 0; dj < cols; ++dj) {
    for (int di = 0; di < rows; ++di) {
      tile[static_cast<std::size_t>(di) + static_cast<std::size_t>(dj) * rows] =
          global_matrix[static_cast<std::size_t>(global_i0 + di) * n + static_cast<std::size_t>(global_j0 + dj)];
    }
  }
}

void unpack_rank_local_matrix_to_root(
    const std::vector<double>& local_matrix,
    std::vector<double>& global_matrix,
    std::size_t n,
    int nb,
    int proc_row,
    int proc_col,
    int nprow,
    int npcol,
    int lld_local) {
  for (int global_i0 = proc_row * nb; global_i0 < static_cast<int>(n); global_i0 += nprow * nb) {
    const int rows = std::min(nb, static_cast<int>(n) - global_i0);
    const int local_i0 = local_index(global_i0, nb, nprow);
    for (int global_j0 = proc_col * nb; global_j0 < static_cast<int>(n); global_j0 += npcol * nb) {
      const int cols = std::min(nb, static_cast<int>(n) - global_j0);
      const int local_j0 = local_index(global_j0, nb, npcol);
      for (int dj = 0; dj < cols; ++dj) {
        for (int di = 0; di < rows; ++di) {
          global_matrix[static_cast<std::size_t>(global_i0 + di) * n + static_cast<std::size_t>(global_j0 + dj)] =
              local_matrix[static_cast<std::size_t>(local_i0 + di) + static_cast<std::size_t>(local_j0 + dj) * lld_local];
        }
      }
    }
  }
}

void distribute_matrix_block_cyclic(
    const std::vector<double>& full_matrix_root,
    std::size_t n,
    int nb,
    int rank,
    int nprow,
    int npcol,
    int myrow,
    int mycol,
    int lld_a,
    std::vector<double>& local_a,
    MPI_Comm comm) {
  const int n_int = static_cast<int>(n);
  if (rank == 0) {
    std::vector<double> tile;
    for (int global_i0 = 0; global_i0 < n_int; global_i0 += nb) {
      const int rows = std::min(nb, n_int - global_i0);
      const int owner_row = block_owner(global_i0, nb, nprow);
      for (int global_j0 = 0; global_j0 < n_int; global_j0 += nb) {
        const int cols = std::min(nb, n_int - global_j0);
        const int owner_col = block_owner(global_j0, nb, npcol);
        const int owner_rank = grid_rank(owner_row, owner_col, npcol);
        if (owner_rank == 0) {
          const int local_i0 = local_index(global_i0, nb, nprow);
          const int local_j0 = local_index(global_j0, nb, npcol);
          copy_global_tile_to_local_column_major(full_matrix_root, n, global_i0, global_j0, rows, cols, local_a, lld_a, local_i0, local_j0);
        } else {
          int header[4] = {global_i0, global_j0, rows, cols};
          MPI_Send(header, 4, MPI_INT, owner_rank, TAG_MATRIX_HEADER, comm);
          pack_column_major_tile_from_global(full_matrix_root, n, global_i0, global_j0, rows, cols, tile);
          MPI_Send(tile.data(), static_cast<int>(tile.size()), MPI_DOUBLE, owner_rank, TAG_MATRIX_TILE, comm);
        }
      }
    }
  } else {
    const int expected = owned_tile_count(n_int, nb, myrow, mycol, nprow, npcol);
    for (int tile_idx = 0; tile_idx < expected; ++tile_idx) {
      int header[4];
      MPI_Recv(header, 4, MPI_INT, 0, TAG_MATRIX_HEADER, comm, MPI_STATUS_IGNORE);
      const int global_i0 = header[0];
      const int global_j0 = header[1];
      const int rows = header[2];
      const int cols = header[3];
      std::vector<double> tile(static_cast<std::size_t>(rows) * cols, 0.0);
      MPI_Recv(tile.data(), static_cast<int>(tile.size()), MPI_DOUBLE, 0, TAG_MATRIX_TILE, comm, MPI_STATUS_IGNORE);
      const int local_i0 = local_index(global_i0, nb, nprow);
      const int local_j0 = local_index(global_j0, nb, npcol);
      for (int dj = 0; dj < cols; ++dj) {
        for (int di = 0; di < rows; ++di) {
          local_a[static_cast<std::size_t>(local_i0 + di) + static_cast<std::size_t>(local_j0 + dj) * lld_a] =
              tile[static_cast<std::size_t>(di) + static_cast<std::size_t>(dj) * rows];
        }
      }
    }
  }
}

void distribute_rhs_block_cyclic(
    const std::vector<double>& rhs_root,
    std::size_t n,
    int nb,
    int rank,
    int nprow,
    int npcol,
    int myrow,
    int mycol,
    int local_rows,
    std::vector<double>& local_b,
    MPI_Comm comm) {
  if (rank == 0) {
    std::vector<std::vector<double>> rank_buffers(static_cast<std::size_t>(nprow));
    for (int prow = 0; prow < nprow; ++prow) {
      const int rows_for_rank = numroc_wrapper(static_cast<int>(n), nb, prow, 0, nprow);
      rank_buffers[static_cast<std::size_t>(prow)].assign(rows_for_rank, 0.0);
    }
    for (int global_i = 0; global_i < static_cast<int>(n); ++global_i) {
      const int owner_row = block_owner(global_i, nb, nprow);
      const int local_i = local_index(global_i, nb, nprow);
      rank_buffers[static_cast<std::size_t>(owner_row)][static_cast<std::size_t>(local_i)] = rhs_root[static_cast<std::size_t>(global_i)];
    }

    for (int prow = 0; prow < nprow; ++prow) {
      const int owner_rank = grid_rank(prow, 0, npcol);
      const int rows_for_rank = static_cast<int>(rank_buffers[static_cast<std::size_t>(prow)].size());
      if (owner_rank == 0) {
        if (mycol == 0 && local_rows > 0) {
          std::copy(rank_buffers[static_cast<std::size_t>(prow)].begin(), rank_buffers[static_cast<std::size_t>(prow)].end(), local_b.begin());
        }
      } else if (rows_for_rank > 0) {
        MPI_Send(rank_buffers[static_cast<std::size_t>(prow)].data(), rows_for_rank, MPI_DOUBLE, owner_rank, TAG_RHS_VECTOR, comm);
      }
    }
  } else if (mycol == 0 && local_rows > 0) {
    MPI_Recv(local_b.data(), local_rows, MPI_DOUBLE, 0, TAG_RHS_VECTOR, comm, MPI_STATUS_IGNORE);
  }
}

double distributed_logdet_from_local_cholesky(
    const std::vector<double>& local_a,
    std::size_t n,
    int nb,
    int myrow,
    int mycol,
    int nprow,
    int npcol,
    int lld_a,
    MPI_Comm comm) {
  double local_sum = 0.0;
  for (int global_i = 0; global_i < static_cast<int>(n); ++global_i) {
    if (block_owner(global_i, nb, nprow) == myrow && block_owner(global_i, nb, npcol) == mycol) {
      const int local_i = local_index(global_i, nb, nprow);
      const int local_j = local_index(global_i, nb, npcol);
      local_sum += std::log(local_a[static_cast<std::size_t>(local_i) + static_cast<std::size_t>(local_j) * lld_a]);
    }
  }
  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
  return 2.0 * global_sum;
}

std::vector<double> gather_alpha_to_root(
    std::size_t n,
    int nb,
    int rank,
    int size,
    int nprow,
    int npcol,
    int myrow,
    int mycol,
    int local_rows,
    const std::vector<double>& local_b,
    MPI_Comm comm) {
  std::vector<double> alpha;
  if (rank == 0) {
    alpha.assign(n, 0.0);
    if (mycol == 0 && local_rows > 0) {
      for (int global_i = 0; global_i < static_cast<int>(n); ++global_i) {
        if (block_owner(global_i, nb, nprow) == myrow) {
          const int local_i = local_index(global_i, nb, nprow);
          alpha[static_cast<std::size_t>(global_i)] = local_b[static_cast<std::size_t>(local_i)];
        }
      }
    }
    for (int r = 1; r < size; ++r) {
      const int proc_row = r / npcol;
      const int proc_col = r % npcol;
      if (proc_col != 0) {
        continue;
      }
      const int rows_for_rank = numroc_wrapper(static_cast<int>(n), nb, proc_row, 0, nprow);
      if (rows_for_rank <= 0) {
        continue;
      }
      std::vector<double> recvbuf(rows_for_rank, 0.0);
      MPI_Recv(recvbuf.data(), rows_for_rank, MPI_DOUBLE, r, TAG_GATHER_ALPHA, comm, MPI_STATUS_IGNORE);
      for (int global_i0 = proc_row * nb; global_i0 < static_cast<int>(n); global_i0 += nprow * nb) {
        const int rows = std::min(nb, static_cast<int>(n) - global_i0);
        const int local_i0 = local_index(global_i0, nb, nprow);
        for (int di = 0; di < rows; ++di) {
          alpha[static_cast<std::size_t>(global_i0 + di)] = recvbuf[static_cast<std::size_t>(local_i0 + di)];
        }
      }
    }
  } else if (mycol == 0 && local_rows > 0) {
    MPI_Send(local_b.data(), local_rows, MPI_DOUBLE, 0, TAG_GATHER_ALPHA, comm);
  }
  return alpha;
}

std::vector<double> gather_local_matrix_to_root(
    std::size_t n,
    int nb,
    int rank,
    int size,
    int nprow,
    int npcol,
    int myrow,
    int mycol,
    int lld_a,
    int local_rows,
    int local_cols,
    const std::vector<double>& local_a,
    MPI_Comm comm) {
  std::vector<double> chol;
  if (rank == 0) {
    chol.assign(n * n, 0.0);
    unpack_rank_local_matrix_to_root(local_a, chol, n, nb, myrow, mycol, nprow, npcol, lld_a);
    for (int r = 1; r < size; ++r) {
      const int proc_row = r / npcol;
      const int proc_col = r % npcol;
      const int rows_for_rank = numroc_wrapper(static_cast<int>(n), nb, proc_row, 0, nprow);
      const int cols_for_rank = numroc_wrapper(static_cast<int>(n), nb, proc_col, 0, npcol);
      const int recv_count = rows_for_rank * cols_for_rank;
      if (recv_count <= 0) {
        continue;
      }
      std::vector<double> recvbuf(static_cast<std::size_t>(recv_count), 0.0);
      MPI_Recv(recvbuf.data(), recv_count, MPI_DOUBLE, r, TAG_GATHER_MATRIX, comm, MPI_STATUS_IGNORE);
      unpack_rank_local_matrix_to_root(recvbuf, chol, n, nb, proc_row, proc_col, nprow, npcol, std::max(1, rows_for_rank));
    }
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i + 1; j < n; ++j) {
        chol[i * n + j] = 0.0;
      }
    }
  } else {
    const int send_count = local_rows * local_cols;
    if (send_count > 0) {
      MPI_Send(local_a.data(), send_count, MPI_DOUBLE, 0, TAG_GATHER_MATRIX, comm);
    }
  }
  return chol;
}

#ifdef HAVE_SCALAPACK
// Milestone 1B: Distributed kernel assembly functions

void broadcast_features(std::vector<double>& x_rated, std::size_t n, int d, int rank, MPI_Comm comm) {
  // Ensure all ranks have the feature matrix
  // Much smaller than broadcasting full kernel (n*d vs n*n)
  const std::size_t total_size = n * static_cast<std::size_t>(d);

  if (rank != 0) {
    x_rated.resize(total_size);
  }

  // Broadcast in chunks if needed (MPI has 2GB limit)
  const std::size_t chunk_size = 100000000;  // ~100M doubles = 800MB per chunk
  std::size_t offset = 0;

  while (offset < total_size) {
    const std::size_t remaining = total_size - offset;
    const int count = static_cast<int>(std::min(remaining, chunk_size));
    MPI_Bcast(x_rated.data() + offset, count, MPI_DOUBLE, 0, comm);
    offset += count;
  }
}

// Determine which global (i,j) blocks this rank owns in block-cyclic distribution
std::vector<std::pair<int, int>> get_owned_blocks(
    int n, int nb, int myrow, int mycol, int nprow, int npcol) {

  std::vector<std::pair<int, int>> blocks;

  // Number of block rows/cols in the matrix
  const int num_block_rows = (n + nb - 1) / nb;
  const int num_block_cols = (n + nb - 1) / nb;

  // Iterate over all block positions
  for (int block_i = 0; block_i < num_block_rows; ++block_i) {
    for (int block_j = 0; block_j < num_block_cols; ++block_j) {
      // Block-cyclic: rank (r,c) owns blocks where (block_i % nprow == r) and (block_j % npcol == c)
      if ((block_i % nprow == myrow) && (block_j % npcol == mycol)) {
        blocks.push_back({block_i, block_j});
      }
    }
  }

  return blocks;
}

// Build local blocks of RBF kernel matrix according to ScaLAPACK layout
std::vector<double> build_local_rbf_blocks_from_features(
    const std::vector<double>& x,
    std::size_t n,
    int d,
    double length_scale,
    double variance,
    double noise,
    int nb,
    int myrow,
    int mycol,
    int nprow,
    int npcol,
    int lld,
    int local_cols) {

  std::cerr << "[DEBUG] Rank (" << myrow << "," << mycol << ") building local blocks: "
            << "n=" << n << " nb=" << nb << " lld=" << lld << " local_cols=" << local_cols << std::endl;

  // Allocate local storage
  std::vector<double> local_matrix(static_cast<std::size_t>(lld) * std::max(1, local_cols), 0.0);

  // Precompute norms
  std::vector<double> norms(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    double sum = 0.0;
    for (int k = 0; k < d; ++k) {
      const double val = x[i * static_cast<std::size_t>(d) + static_cast<std::size_t>(k)];
      sum += val * val;
    }
    norms[i] = sum;
  }

  // Get blocks owned by this rank
  auto owned_blocks = get_owned_blocks(static_cast<int>(n), nb, myrow, mycol, nprow, npcol);

  std::cerr << "[DEBUG] Rank (" << myrow << "," << mycol << ") owns " << owned_blocks.size() << " blocks" << std::endl;

  const double inv_two_ell_sq = -0.5 / (length_scale * length_scale);

  // Process each owned block
  for (const auto& block_pair : owned_blocks) {
    const int block_i = block_pair.first;
    const int block_j = block_pair.second;

    // Global indices for this block
    const int global_i_start = block_i * nb;
    const int global_j_start = block_j * nb;
    const int block_rows = std::min(nb, static_cast<int>(n) - global_i_start);
    const int block_cols = std::min(nb, static_cast<int>(n) - global_j_start);

    // Local column index for this block
    const int local_col_start = (block_j / npcol) * nb;

    // Compute kernel values for this block
    for (int local_i = 0; local_i < block_rows; ++local_i) {
      const int global_i = global_i_start + local_i;

      // Local row index (accounting for block-cyclic distribution)
      const int local_row_start = (block_i / nprow) * nb;
      const int local_row = local_row_start + local_i;

      for (int local_j = 0; local_j < block_cols; ++local_j) {
        const int global_j = global_j_start + local_j;
        const int local_col = local_col_start + local_j;

        // Compute dot product
        double dot = 0.0;
        for (int k = 0; k < d; ++k) {
          dot += x[global_i * static_cast<std::size_t>(d) + k] *
                 x[global_j * static_cast<std::size_t>(d) + k];
        }

        // RBF kernel: exp(-0.5 * ||x_i - x_j||^2 / length_scale^2)
        double d2 = norms[global_i] + norms[global_j] - 2.0 * dot;
        if (d2 < 0.0) d2 = 0.0;

        double value = variance * std::exp(inv_two_ell_sq * d2);

        // Add noise to diagonal
        if (global_i == global_j) {
          value += noise * noise;
        }

        // Store in local matrix
        local_matrix[local_row + static_cast<std::size_t>(local_col) * lld] = value;
      }
    }
  }

  std::cerr << "[DEBUG] Rank (" << myrow << "," << mycol << ") completed local kernel assembly" << std::endl;
  std::cerr.flush();

  // Debug: Check first few values
  std::cerr << "[DEBUG] Checking matrix values for rank (" << myrow << "," << mycol << ")..." << std::endl;
  std::cerr.flush();

  if (local_matrix.size() >= 1) {
    std::cerr << "[DEBUG] local_matrix.size() = " << local_matrix.size() << std::endl;
    std::cerr << "[DEBUG] lld=" << lld << " local_cols=" << local_cols << std::endl;
    std::cerr.flush();

    // Print first 5 diagonal elements
    const int n_diag = std::min(5, std::min(lld, local_cols));
    for (int i = 0; i < n_diag; ++i) {
      const size_t idx = i + static_cast<std::size_t>(i) * lld;
      std::cerr << "[DEBUG]   local[" << i << "," << i << "] at idx=" << idx
                << " = " << local_matrix[idx] << std::endl;
      std::cerr.flush();
    }

    // Print first off-diagonal if it exists
    if (local_cols > 1) {
      const size_t idx = 0 + static_cast<std::size_t>(1) * lld;
      std::cerr << "[DEBUG]   local[0,1] at idx=" << idx
                << " = " << local_matrix[idx] << std::endl;
      std::cerr.flush();
    }
  }

  std::cerr << "[DEBUG] Matrix value check complete" << std::endl;
  std::cerr.flush();

  return local_matrix;
}
#endif

NativeResult run_scalapack(std::size_t n, int rank, int size, const std::vector<double>& full_matrix_root, const std::vector<double>& rhs_root, int block_size, MPI_Comm comm) {
  NativeResult result;
  result.backend = "scalapack";
  result.implemented = true;
  result.message = "Distributed ScaLAPACK Cholesky factorization and solve with direct block-cyclic distribution and local-to-root reconstruction of outputs.";

  const int n_int = static_cast<int>(n);
  const int nb = std::max(1, block_size);
  const int nprow = choose_nprow(size);
  const int npcol = size / nprow;

  // Initialize BLACS-MPI interface
  int blacs_rank = -1;
  int blacs_size = -1;
  Cblacs_pinfo(&blacs_rank, &blacs_size);

  int ictxt = 0;
  // Get BLACS context from MPI communicator (use 0 instead of -1)
  Cblacs_get(0, 0, &ictxt);
  Cblacs_gridinit(&ictxt, "R", nprow, npcol);
  int myrow = -1;
  int mycol = -1;
  int grid_rows = 0;
  int grid_cols = 0;
  Cblacs_gridinfo(ictxt, &grid_rows, &grid_cols, &myrow, &mycol);

  const int rsrc = 0;
  const int csrc = 0;
  const int nrhs = 1;
  const int local_rows = numroc_wrapper(n_int, nb, myrow, rsrc, grid_rows);
  const int local_cols = numroc_wrapper(n_int, nb, mycol, csrc, grid_cols);
  const int local_rhs_cols = numroc_wrapper(nrhs, nb, mycol, csrc, grid_cols);
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

  const auto total_start = std::chrono::steady_clock::now();
  distribute_matrix_block_cyclic(full_matrix_root, n, nb, rank, grid_rows, grid_cols, myrow, mycol, lld_a, local_a, comm);
  distribute_rhs_block_cyclic(rhs_root, n, nb, rank, grid_rows, grid_cols, myrow, mycol, local_rows, local_b, comm);
  MPI_Barrier(comm);

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
  result.logdet = distributed_logdet_from_local_cholesky(local_a, n, nb, myrow, mycol, grid_rows, grid_cols, lld_a, comm);
  MPI_Barrier(comm);
  const auto solve_end = std::chrono::steady_clock::now();

  const auto gather_start = std::chrono::steady_clock::now();
  gather_alpha_to_root(n, nb, rank, size, grid_rows, grid_cols, myrow, mycol, local_rows, local_b, comm).swap(result.alpha);
  gather_local_matrix_to_root(n, nb, rank, size, grid_rows, grid_cols, myrow, mycol, lld_a, local_rows, local_cols, local_a, comm).swap(result.chol);
  MPI_Barrier(comm);
  const auto gather_end = std::chrono::steady_clock::now();
  const auto total_end = std::chrono::steady_clock::now();

  Cblacs_gridexit(ictxt);

  if (rank == 0) {
    result.factor_seconds = std::chrono::duration<double>(factor_end - factor_start).count();
    result.solve_seconds = std::chrono::duration<double>(solve_end - solve_start).count();
    result.gather_seconds = std::chrono::duration<double>(gather_end - gather_start).count();
    result.total_seconds = std::chrono::duration<double>(total_end - total_start).count();
  }
  return result;
}

// Milestone 1B: Distributed kernel assembly version
NativeResult run_scalapack_distributed(
    std::size_t n,
    int rank,
    int size,
    const std::vector<double>& x_rated_root,  // Features on root (n×d)
    const std::vector<double>& rhs_root,
    int d,
    double length_scale,
    double variance,
    double noise,
    int block_size,
    MPI_Comm comm) {

  NativeResult result;
  result.backend = "scalapack_distributed";
  result.implemented = true;
  result.message = "Distributed ScaLAPACK with parallel kernel assembly from features (Milestone 1B).";

  const int n_int = static_cast<int>(n);
  const int nb = std::max(1, block_size);
  const int nprow = choose_nprow(size);
  const int npcol = size / nprow;

  std::cerr << "[DEBUG GRID] size=" << size << " nprow=" << nprow << " npcol=" << npcol << std::endl;

  // Initialize BLACS-MPI interface
  int blacs_rank = -1;
  int blacs_size = -1;
  Cblacs_pinfo(&blacs_rank, &blacs_size);
  std::cerr << "[DEBUG GRID] After Cblacs_pinfo: blacs_rank=" << blacs_rank << " blacs_size=" << blacs_size << std::endl;

  int ictxt = 0;
  std::cerr << "[DEBUG GRID] ictxt before Cblacs_get: " << ictxt << std::endl;

  // Get BLACS context from MPI communicator (use 0 instead of -1)
  Cblacs_get(0, 0, &ictxt);
  std::cerr << "[DEBUG GRID] ictxt after Cblacs_get: " << ictxt << std::endl;

  Cblacs_gridinit(&ictxt, "R", nprow, npcol);
  std::cerr << "[DEBUG GRID] ictxt after Cblacs_gridinit: " << ictxt << std::endl;

  int myrow = -1;
  int mycol = -1;
  int grid_rows = 0;
  int grid_cols = 0;
  Cblacs_gridinfo(ictxt, &grid_rows, &grid_cols, &myrow, &mycol);

  std::cerr << "[DEBUG GRID] After gridinfo: grid_rows=" << grid_rows << " grid_cols=" << grid_cols
            << " myrow=" << myrow << " mycol=" << mycol << std::endl;

  const int rsrc = 0;
  const int csrc = 0;
  const int nrhs = 1;
  const int local_rows = numroc_wrapper(n_int, nb, myrow, rsrc, grid_rows);
  const int local_cols = numroc_wrapper(n_int, nb, mycol, csrc, grid_cols);
  const int local_rhs_cols = numroc_wrapper(nrhs, nb, mycol, csrc, grid_cols);
  const int lld_a = std::max(1, local_rows);
  const int lld_b = std::max(1, local_rows);

  std::vector<double> local_b(static_cast<std::size_t>(lld_b) * std::max(1, local_rhs_cols), 0.0);

  int desc_a[9];
  int desc_b[9];
  int info_desc_a = 0;
  int info_desc_b = 0;

  std::cerr << "[DEBUG DESC] Before descinit_: ictxt=" << ictxt << " n_int=" << n_int
            << " nb=" << nb << " rsrc=" << rsrc << " csrc=" << csrc
            << " lld_a=" << lld_a << " lld_b=" << lld_b << std::endl;

  descinit_(desc_a, &n_int, &n_int, &nb, &nb, &rsrc, &csrc, &ictxt, &lld_a, &info_desc_a);
  descinit_(desc_b, &n_int, &nrhs, &nb, &nb, &rsrc, &csrc, &ictxt, &lld_b, &info_desc_b);

  std::cerr << "[DEBUG DESC] After descinit_: info_desc_a=" << info_desc_a << " info_desc_b=" << info_desc_b << std::endl;
  std::cerr << "[DEBUG DESC] desc_a=[" << desc_a[0] << "," << desc_a[1] << "," << desc_a[2]
            << "," << desc_a[3] << "," << desc_a[4] << "," << desc_a[5]
            << "," << desc_a[6] << "," << desc_a[7] << "," << desc_a[8] << "]" << std::endl;

  if (info_desc_a != 0 || info_desc_b != 0) {
    result.implemented = false;
    result.message = "ScaLAPACK descriptor initialization failed.";
    Cblacs_gridexit(ictxt);
    return result;
  }

  const auto total_start = std::chrono::steady_clock::now();

  // Step 1: Broadcast features to all ranks (much smaller than full matrix!)
  std::vector<double> x_rated = x_rated_root;  // Copy on root, empty on others
  const auto broadcast_start = std::chrono::steady_clock::now();
  broadcast_features(x_rated, n, d, rank, comm);
  const auto broadcast_end = std::chrono::steady_clock::now();

  if (rank == 0) {
    std::cerr << "[Milestone 1B] Broadcast " << (n*d*8.0/1e6) << " MB features in "
              << std::chrono::duration<double>(broadcast_end - broadcast_start).count()
              << " seconds" << std::endl;
  }

  // Step 2: Each rank builds its local blocks (PARALLEL!)
  const auto assembly_start = std::chrono::steady_clock::now();
  std::vector<double> local_a = build_local_rbf_blocks_from_features(
      x_rated, n, d, length_scale, variance, noise,
      nb, myrow, mycol, nprow, npcol, lld_a, local_cols);
  const auto assembly_end = std::chrono::steady_clock::now();

  if (rank == 0) {
    std::cerr << "[Milestone 1B] Parallel kernel assembly in "
              << std::chrono::duration<double>(assembly_end - assembly_start).count()
              << " seconds" << std::endl;
  }

  // Step 3: Distribute RHS (still needed, but much smaller than matrix)
  std::cerr << "[DEBUG] Rank " << rank << " about to distribute RHS, local_rows=" << local_rows << std::endl;
  std::cerr.flush();

  distribute_rhs_block_cyclic(rhs_root, n, nb, rank, grid_rows, grid_cols, myrow, mycol, local_rows, local_b, comm);

  std::cerr << "[DEBUG] Rank " << rank << " RHS distributed" << std::endl;
  std::cerr.flush();

  MPI_Barrier(comm);

  std::cerr << "[DEBUG] Rank " << rank << " passed barrier" << std::endl;
  std::cerr.flush();

  // Step 4: Cholesky factorization (same as before)
  const auto factor_start = std::chrono::steady_clock::now();
  const char uplo = 'L';
  const int ia = 1;
  const int ja = 1;

  // Sanity check: verify local_a is populated and has reasonable values
  if (rank == 0) {
    double min_val = 1e100;
    double max_val = -1e100;
    int nan_count = 0;
    for (size_t i = 0; i < local_a.size(); ++i) {
      if (std::isnan(local_a[i])) {
        nan_count++;
      } else {
        min_val = std::min(min_val, local_a[i]);
        max_val = std::max(max_val, local_a[i]);
      }
    }
    std::cerr << "[SANITY CHECK] local_a: size=" << local_a.size()
              << " min=" << min_val << " max=" << max_val
              << " NaNs=" << nan_count << std::endl;

    // Check for all zeros (sign of assembly failure)
    int zero_count = 0;
    for (size_t i = 0; i < std::min(size_t(100), local_a.size()); ++i) {
      if (local_a[i] == 0.0) zero_count++;
    }
    std::cerr << "[SANITY CHECK] First 100 elements: " << zero_count << " zeros" << std::endl;
  }

  std::cerr << "[DEBUG] About to call pdpotrf_: n=" << n_int << " nb=" << nb
            << " desc_a=[" << desc_a[0] << "," << desc_a[1] << "," << desc_a[2]
            << "," << desc_a[3] << "," << desc_a[4] << "," << desc_a[5]
            << "," << desc_a[6] << "," << desc_a[7] << "," << desc_a[8] << "]" << std::endl;
  std::cerr << "[DEBUG] local_a.size()=" << local_a.size() << " local_a.data()=" << (void*)local_a.data()
            << " lld_a=" << lld_a << " local_cols=" << local_cols << std::endl;
  std::cerr.flush();

  pdpotrf_(&uplo, &n_int, local_a.data(), &ia, &ja, desc_a, &result.info_potrf);

  std::cerr << "[DEBUG] pdpotrf_ returned with info=" << result.info_potrf << std::endl;
  std::cerr.flush();

  MPI_Barrier(comm);
  const auto factor_end = std::chrono::steady_clock::now();

  // Step 5: Solve (same as before)
  const auto solve_start = std::chrono::steady_clock::now();
  if (result.info_potrf == 0) {
    const int ib = 1;
    const int jb = 1;
    pdpotrs_(&uplo, &n_int, &nrhs, local_a.data(), &ia, &ja, desc_a, local_b.data(), &ib, &jb, desc_b, &result.info_potrs);
  }
  result.logdet = distributed_logdet_from_local_cholesky(local_a, n, nb, myrow, mycol, grid_rows, grid_cols, lld_a, comm);
  MPI_Barrier(comm);
  const auto solve_end = std::chrono::steady_clock::now();

  // Step 6: Gather results (same as before)
  const auto gather_start = std::chrono::steady_clock::now();
  gather_alpha_to_root(n, nb, rank, size, grid_rows, grid_cols, myrow, mycol, local_rows, local_b, comm).swap(result.alpha);
  gather_local_matrix_to_root(n, nb, rank, size, grid_rows, grid_cols, myrow, mycol, lld_a, local_rows, local_cols, local_a, comm).swap(result.chol);
  MPI_Barrier(comm);
  const auto gather_end = std::chrono::steady_clock::now();
  const auto total_end = std::chrono::steady_clock::now();

  Cblacs_gridexit(ictxt);

  if (rank == 0) {
    result.factor_seconds = std::chrono::duration<double>(factor_end - factor_start).count();
    result.solve_seconds = std::chrono::duration<double>(solve_end - solve_start).count();
    result.gather_seconds = std::chrono::duration<double>(gather_end - gather_start).count();
    result.total_seconds = std::chrono::duration<double>(total_end - total_start).count();

    std::cerr << "[Milestone 1B] Timings:" << std::endl;
    std::cerr << "  Broadcast: " << std::chrono::duration<double>(broadcast_end - broadcast_start).count() << "s" << std::endl;
    std::cerr << "  Assembly:  " << std::chrono::duration<double>(assembly_end - assembly_start).count() << "s" << std::endl;
    std::cerr << "  Factor:    " << result.factor_seconds << "s" << std::endl;
    std::cerr << "  Solve:     " << result.solve_seconds << "s" << std::endl;
    std::cerr << "  Gather:    " << result.gather_seconds << "s" << std::endl;
    std::cerr << "  Total:     " << result.total_seconds << "s" << std::endl;
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
