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

#include <cuml/tsa/arima_common.h>

#define ARRAY_SPEC(name, type, length)     \
  {                                        \
    (name) = (type*)(mem_buffer + offset); \
    offset += (length) * sizeof(type);     \
  }

namespace ML {

template <typename DataT>
class ARIMAMemory {
 public:
  // These public attributes are used to access the managed arrays
  DataT *Z, *T, *P, *R, *RRT, *tmp_r, *tmp_r2, *tmp_r4, *tmp_r4_bis;
  DataT **Z_members, **T_members, **P_members, **R_members, **RRT_members,
    **tmp_r_members, **tmp_r2_members, **tmp_r4_members, **tmp_r4_bis_members;
  int *perm, *info;

  ARIMAMemory(uint8_t* ptr, const ARIMAOrder& order, int n_obs, int batch_size)
    : mem_buffer(ptr), order(order), n_obs(n_obs), batch_size(batch_size) {
    int r = order.r();
    int r2 = r * r;
    int r4 = r2 * r2;

    int offset = 0;

    // Any update here must be reflected in compute_buffer_size!
    ARRAY_SPEC(Z, DataT, batch_size * r);
    ARRAY_SPEC(T, DataT, batch_size * r2);
    ARRAY_SPEC(P, DataT, batch_size * r2);
    ARRAY_SPEC(R, DataT, batch_size * r);
    ARRAY_SPEC(RRT, DataT, batch_size * r2);
    /// TODO: remove if can be replaced by tmp_r2 and tmp_4
    ARRAY_SPEC(tmp_r, DataT, batch_size * r);
    ARRAY_SPEC(tmp_r2, DataT, batch_size * r2);
    ARRAY_SPEC(tmp_r4, DataT, batch_size * r4);
    ARRAY_SPEC(tmp_r4_bis, DataT, batch_size * r4);

    ARRAY_SPEC(Z_members, DataT*, batch_size);
    ARRAY_SPEC(T_members, DataT*, batch_size);
    ARRAY_SPEC(P_members, DataT*, batch_size);
    ARRAY_SPEC(R_members, DataT*, batch_size);
    ARRAY_SPEC(RRT_members, DataT*, batch_size);
    ARRAY_SPEC(tmp_r_members, DataT*, batch_size);
    ARRAY_SPEC(tmp_r2_members, DataT*, batch_size);
    ARRAY_SPEC(tmp_r4_members, DataT*, batch_size);
    ARRAY_SPEC(tmp_r4_bis_members, DataT*, batch_size);

    ARRAY_SPEC(perm, int, batch_size* r2);
    ARRAY_SPEC(info, int, batch_size);

    buffer_size = offset;

    // To prevent mismatch between allocated and used memory
    ASSERT(buffer_size == compute_buffer_size(order, n_obs, batch_size),
           "ARIMA buffer size computation is wrong");
  }

  static int compute_buffer_size(const ARIMAOrder& order, int n_obs,
                                 int batch_size) {
    int r = order.r();
    int r2 = r * r;
    int r4 = r2 * r2;

    return batch_size * (2 * r4 + 4 * r2 + 3 * r) * sizeof(DataT) +
           9 * batch_size * sizeof(DataT*) +
           (r2 + 1) * batch_size * sizeof(int);
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
