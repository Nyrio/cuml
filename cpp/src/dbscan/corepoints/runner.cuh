/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#pragma once

#include <common/cumlHandle.hpp>
#include "algo.cuh"
#include "pack.h"

namespace ML {
namespace Dbscan {
namespace CorePoints {

/**
 * Compute the core points from the vertex degrees and min_pts criterion
 * @param[in]  handle          cuML handle
 * @param[in]  vd              Vertex degrees
 * @param[out] mask            Boolean core point mask
 * @param[in]  min_pts         Core point criterion
 * @param[in]  start_vertex_id First point of the batch
 * @param[in]  batch_size      Batch size
 * @param[in]  stream          CUDA stream
 */
template <typename Index_ = int>
void run(const raft::handle_t& handle, const Index_* vd, bool* mask,
         Index_ min_pts, Index_ start_vertex_id, Index_ batch_size,
         cudaStream_t stream) {
  Pack<Index_> data = {vd, mask, min_pts};
  Algo::launcher<Index_>(handle, data, start_vertex_id, batch_size, stream);
}

}  // namespace CorePoints
}  // namespace Dbscan
}  // namespace ML
