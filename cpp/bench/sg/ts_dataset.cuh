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

#include <cuda_utils.h>
#include <common/cumlHandle.hpp>
#include <cuml/cuml.hpp>

#include <cuml/tsa/arima_common.h>
#include <random/make_arima.h>

namespace ML {
namespace Bench {

/** General information about a time series dataset */
struct TimeSeriesParams {
  int batch_size;
  int n_obs;
  uint64_t seed;
};

struct TimeSeriesScale {
  double series;
  double noise;
  double intercept;
};

/**
 * @brief A simple object to hold the loaded dataset for benchmarking
 * @tparam D type of the time series data
 */
template <typename D>
struct TimeSeriesDataset {
  /** input data */
  D* X;

  /** allocate space needed for the dataset */
  void allocate(const cumlHandle& handle, const TimeSeriesParams& p) {
    auto allocator = handle.getDeviceAllocator();
    auto stream = handle.getStream();
    X = (D*)allocator->allocate(p.batch_size * p.n_obs * sizeof(D), stream);
  }

  /** free-up the buffers */
  void deallocate(const cumlHandle& handle, const TimeSeriesParams& p) {
    auto allocator = handle.getDeviceAllocator();
    auto stream = handle.getStream();
    allocator->deallocate(X, p.batch_size * p.n_obs * sizeof(D), stream);
  }

  /**
   * Generate random time series for a given ARIMA order.
   * Assumes that the user has already called `allocate`
   */
  void arima(const cumlHandle& handle, const TimeSeriesParams& p,
             const TimeSeriesScale scale, const ARIMAOrder& order) {
    auto stream = handle.getStream();
    auto allocator = handle.getDeviceAllocator();

    MLCommon::Random::make_arima<D>(X, p.batch_size, p.n_obs, order, allocator,
                                    stream, (D)scale.series, (D)scale.noise,
                                    (D)scale.intercept, nullptr, nullptr,
                                    nullptr, nullptr, nullptr, p.seed);
  }
};

}  // namespace Bench
}  // namespace ML
