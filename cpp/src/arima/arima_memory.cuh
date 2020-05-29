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

#pragma once

#include <cuda_runtime.h>
#include <cstddef>

#include <cuml/tsa/arima_common.h>
#include <cuml/common/cuml_allocator.hpp>
#include <cuml/cuml.hpp>

namespace ML {

template <typename DataT>
class ARIMAMemory {
 public:
  // These public attributes are used to access the managed arrays
  DataT *Z, *T, *P, *R, *RRT;

  ARIMAMemory(uint8_t* ptr, const ARIMAOrder& order, int n_obs, int batch_size)
    : mem_buffer(ptr),
      mem_buffer(mem_buffer),
      order(order),
      n_obs(n_obs),
      batch_size(batch_size) {
    int r = order.r();
    int r2 = r * r;

    int offset = 0;
  
    // Any update here must be reflected in compute_buffer_size!
    Z = (DataT*)mem_buffer;
    offset += batch_size * r;
    T = (DataT*)(mem_buffer + offset);
    offset += batch_size * r2;
    P = (DataT*)(mem_buffer + offset);
    offset += batch_size * r2;
    R = (DataT*)(mem_buffer + offset);
    offset += batch_size * r;
    RRT = (DataT*)(mem_buffer + offset);
    offset += batch_size * r2;
  
    buffer_size = offset;
  }

  static int compute_buffer_size(const ARIMAOrder& order, int n_obs,
                                 int batch_size) {
    int r = order.r();
    int r2 = r * r;

    return batch_size * (3 * r2 + 2 * r);
  }

 protected:
  // Internal attributes
  uint8_t* mem_buffer;
  ARIMAOrder order;
  int batch_size;
  int n_obs;
  int buffer_size;
};

}  // namespace ML
