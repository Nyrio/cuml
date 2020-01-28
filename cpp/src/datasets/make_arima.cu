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

#include <common/cumlHandle.hpp>
#include <cuml/datasets/make_arima.hpp>
#include "random/make_arima.h"

namespace ML {
namespace Datasets {

template <typename DataT, typename IdxT>
inline void make_arima_helper(const cumlHandle& handle, DataT* out,
                              IdxT batch_size, IdxT n_obs, ARIMAOrder order,
                              DataT scale, DataT noise_scale, DataT diff_scale,
                              DataT* d_mu, DataT* d_ar, DataT* d_ma,
                              DataT* d_sar, DataT* d_sma, uint64_t seed) {
  auto stream = handle.getStream();
  auto allocator = handle.getImpl().getDeviceAllocator();

  MLCommon::Random::make_arima(out, batch_size, n_obs, order, allocator, stream,
                               scale, noise_scale, diff_scale, d_mu, d_ar, d_ma,
                               d_sar, d_sma, seed);
}

void make_arima(const cumlHandle& handle, float* out, int batch_size, int n_obs,
                ARIMAOrder order, float scale, float noise_scale,
                float diff_scale, float* d_mu, float* d_ar, float* d_ma,
                float* d_sar, float* d_sma, uint64_t seed) {
  make_arima_helper(handle, out, batch_size, n_obs, order, scale, noise_scale,
                    diff_scale, d_mu, d_ar, d_ma, d_sar, d_sma, seed);
}

void make_arima(const cumlHandle& handle, double* out, int batch_size,
                int n_obs, ARIMAOrder order, double scale, double noise_scale,
                double diff_scale, double* d_mu, double* d_ar, double* d_ma,
                double* d_sar, double* d_sma, uint64_t seed) {
  make_arima_helper(handle, out, batch_size, n_obs, order, scale, noise_scale,
                    diff_scale, d_mu, d_ar, d_ma, d_sar, d_sma, seed);
}

}  // namespace Datasets
}  // namespace ML