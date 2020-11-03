/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cuml/cuml_api.h>
#include <common/cumlHandle.hpp>
#include <cuml/cluster/dbscan_mg.hpp>
#include "dbscan.cuh"

namespace ML {
namespace Dbscan {
namespace opg {

void fit(const raft::handle_t &handle, float *input, int n_rows, int n_cols,
         float eps, int min_pts, int *labels, size_t max_bytes_per_batch,
         int verbosity) {
  dbscanFitImpl<float, int, true>(handle, input, n_rows, n_cols, eps, min_pts,
                                  labels, nullptr, max_bytes_per_batch,
                                  handle.get_stream(), verbosity);
}

void fit(const raft::handle_t &handle, double *input, int n_rows, int n_cols,
         double eps, int min_pts, int *labels, size_t max_bytes_per_batch,
         int verbosity) {
  dbscanFitImpl<double, int, true>(handle, input, n_rows, n_cols, eps, min_pts,
                                   labels, nullptr, max_bytes_per_batch,
                                   handle.get_stream(), verbosity);
}

void fit(const raft::handle_t &handle, float *input, int n_rows, int n_cols,
         float eps, int min_pts, int *labels, int *core_sample_indices,
         size_t max_bytes_per_batch, int verbosity) {
  dbscanFitImpl<float, int, true>(
    handle, input, n_rows, n_cols, eps, min_pts, labels, core_sample_indices,
    max_bytes_per_batch, handle.get_stream(), verbosity);
}

void fit(const raft::handle_t &handle, double *input, int n_rows, int n_cols,
         double eps, int min_pts, int *labels, int *core_sample_indices,
         size_t max_bytes_per_batch, int verbosity) {
  dbscanFitImpl<double, int, true>(
    handle, input, n_rows, n_cols, eps, min_pts, labels, core_sample_indices,
    max_bytes_per_batch, handle.get_stream(), verbosity);
}

void fit(const raft::handle_t &handle, float *input, int64_t n_rows,
         int64_t n_cols, float eps, int min_pts, int64_t *labels,
         size_t max_bytes_per_batch, int verbosity) {
  dbscanFitImpl<float, int64_t, true>(
    handle, input, n_rows, n_cols, eps, min_pts, labels, nullptr,
    max_bytes_per_batch, handle.get_stream(), verbosity);
}

void fit(const raft::handle_t &handle, double *input, int64_t n_rows,
         int64_t n_cols, double eps, int min_pts, int64_t *labels,
         size_t max_bytes_per_batch, int verbosity) {
  dbscanFitImpl<double, int64_t, true>(
    handle, input, n_rows, n_cols, eps, min_pts, labels, nullptr,
    max_bytes_per_batch, handle.get_stream(), verbosity);
}

void fit(const raft::handle_t &handle, float *input, int64_t n_rows,
         int64_t n_cols, float eps, int min_pts, int64_t *labels,
         int64_t *core_sample_indices, size_t max_bytes_per_batch,
         int verbosity) {
  dbscanFitImpl<float, int64_t>(
    handle, input, n_rows, n_cols, eps, min_pts, labels, core_sample_indices,
    max_bytes_per_batch, handle.get_stream(), verbosity);
}

void fit(const raft::handle_t &handle, double *input, int64_t n_rows,
         int64_t n_cols, double eps, int min_pts, int64_t *labels,
         int64_t *core_sample_indices, size_t max_bytes_per_batch,
         int verbosity) {
  dbscanFitImpl<double, int64_t>(
    handle, input, n_rows, n_cols, eps, min_pts, labels, core_sample_indices,
    max_bytes_per_batch, handle.get_stream(), verbosity);
}

}  // namespace opg
}  // namespace Dbscan
}  // end namespace ML