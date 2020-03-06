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
                ARIMAOrder order, float scale = 1.0f, float noise_scale = 0.2f,
                float intercept_scale = 1.0f,
                ARIMAParams<float> params = ARIMAParams<float>(
                  nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
                uint64_t seed = 0ULL);

void make_arima(const cumlHandle& handle, double* out, int batch_size,
                int n_obs, ARIMAOrder order, double scale = 1.0,
                double noise_scale = 0.2, double intercept_scale = 1.0,
                ARIMAParams<double> params = ARIMAParams<double>(
                  nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
                uint64_t seed = 0ULL);

}  // namespace Datasets
}  // namespace ML