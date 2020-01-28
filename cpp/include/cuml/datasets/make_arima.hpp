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
#include <cuml/cuml.hpp>

namespace ML {
namespace Datasets {

/**
 * @todo: docs 
 */
void make_arima(const cumlHandle& handle, float* out, int batch_size, int n_obs,
                ARIMAOrder order, float scale = 1.0f, float noise_scale = 0.1f,
                float diff_scale = 0.3f, float* d_mu = nullptr,
                float* d_ar = nullptr, float* d_ma = nullptr,
                float* d_sar = nullptr, float* d_sma = nullptr,
                uint64_t seed = 0ULL);

void make_arima(const cumlHandle& handle, double* out, int batch_size,
                int n_obs, ARIMAOrder order, double scale = 1.0,
                double noise_scale = 0.1, double diff_scale = 0.3,
                double* d_mu = nullptr, double* d_ar = nullptr,
                double* d_ma = nullptr, double* d_sar = nullptr,
                double* d_sma = nullptr, uint64_t seed = 0ULL);

}  // namespace Datasets
}  // namespace ML
