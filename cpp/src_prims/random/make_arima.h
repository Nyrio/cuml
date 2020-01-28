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

#include <random>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include "cuml/common/cuml_allocator.hpp"
#include "cuml/tsa/arima_common.h"
#include "rng.h"
#include "timeSeries/arima_helpers.h"
#include "utils.h"

namespace MLCommon {
namespace Random {

/**
 * @brief Time series generator for a given ARIMA order
 *
 * @tparam  DataT  Scalar type
 * @todo: docs (+ modify interface?) + way to return params
 *                                     (as coef in make_regression)
 */
template <typename DataT>
void make_arima(DataT* out, int batch_size, int n_obs, ML::ARIMAOrder order,
                std::shared_ptr<deviceAllocator> allocator, cudaStream_t stream,
                DataT scale = (DataT)1.0, DataT noise_scale = (DataT)0.1,
                DataT diff_scale = (DataT)0.3, DataT* d_mu = nullptr,
                DataT* d_ar = nullptr, DataT* d_ma = nullptr,
                DataT* d_sar = nullptr, DataT* d_sma = nullptr,
                uint64_t seed = 0ULL, GeneratorType type = GenPhilox) {
  int d_sD = order.d + order.s * order.D;
  int p_sP = order.p + order.s * order.P;
  int q_sQ = order.q + order.s * order.Q;
  auto counting = thrust::make_counting_iterator(0);

  DataT intercept_scale = d_sD ? diff_scale : scale;

  // Create random generators and distributions
  std::default_random_engine cpu_gen(seed);
  Rng gpu_gen(seed, type);
  std::uniform_real_distribution<DataT> udis((DataT)0.0, (DataT)1.0);

  // Generate parameters
  device_buffer<DataT> mu(allocator, stream);
  device_buffer<DataT> ar(allocator, stream);
  device_buffer<DataT> ma(allocator, stream);
  device_buffer<DataT> sar(allocator, stream);
  device_buffer<DataT> sma(allocator, stream);
  if (order.k) {
    if (d_mu == nullptr) {
      mu.resize(batch_size, stream);
      d_mu = mu.data();
    }
    gpu_gen.uniform(d_mu, batch_size, -intercept_scale, intercept_scale,
                    stream);
  }
  if (order.p) {
    if (d_ar == nullptr) {
      ar.resize(batch_size * order.p, stream);
      d_ar = ar.data();
    }
    gpu_gen.uniform(d_ar, batch_size * order.p, (DataT)-1.0, (DataT)1.0,
                    stream);
  }
  if (order.q) {
    if (d_ma == nullptr) {
      ma.resize(batch_size * order.q, stream);
      d_ma = ma.data();
    }
    gpu_gen.uniform(d_ma, batch_size * order.q, (DataT)-1.0, (DataT)1.0,
                    stream);
  }
  if (order.P) {
    if (d_sar == nullptr) {
      sar.resize(batch_size * order.P, stream);
      d_sar = sar.data();
    }
    gpu_gen.uniform(d_sar, batch_size * order.P, (DataT)-1.0, (DataT)1.0,
                    stream);
  }
  if (order.Q) {
    if (d_sma == nullptr) {
      sma.resize(batch_size * order.Q, stream);
      d_sma = sma.data();
    }
    gpu_gen.uniform(d_sma, batch_size * order.Q, (DataT)-1.0, (DataT)1.0,
                    stream);
  }

  // Create coefficient vectors for the AR+SAR and MA+SMA components
  device_buffer<DataT> ar_vec(allocator, stream);
  device_buffer<DataT> ma_vec(allocator, stream);
  ar_vec.resize(batch_size * p_sP, stream);
  ma_vec.resize(batch_size * q_sQ, stream);
  DataT* d_ar_vec = ar_vec.data();
  DataT* d_ma_vec = ar_vec.data();
  if (p_sP) {
    thrust::for_each(thrust::cuda::par.on(stream), counting,
                     counting + batch_size, [=] __device__(int ib) {
                       DataT* b_ar_vec = d_ar_vec + ib * p_sP;
                       for (int ip = 0; ip < p_sP; ip++) {
                         b_ar_vec[ip] = TimeSeries::reduced_polynomial<true>(
                           ib, d_ar, order.p, d_sar, order.P, order.s, ip + 1);
                       }
                     });
  }
  if (q_sQ) {
    thrust::for_each(thrust::cuda::par.on(stream), counting,
                     counting + batch_size, [=] __device__(int ib) {
                       DataT* b_ma_vec = d_ma_vec + ib * q_sQ;
                       for (int iq = 0; iq < q_sQ; iq++) {
                         b_ma_vec[iq] = TimeSeries::reduced_polynomial<true>(
                           ib, d_ma, order.q, d_sar, order.Q, order.s, iq + 1);
                       }
                     });
  }

  // Generate d+s*D starting values per series
  device_buffer<DataT> starting_values(allocator, stream);
  if (d_sD) {
    starting_values.resize(batch_size * d_sD, stream);
    DataT mean = udis(cpu_gen);
    gpu_gen.uniform(starting_values.data(), batch_size * d_sD, mean - scale,
                    mean + scale, stream);
  }

  // Create buffer for differenced series
  DataT* d_diff;
  device_buffer<DataT> diff_data(allocator, stream);
  if (d_sD) {
    diff_data.resize(batch_size * (n_obs - d_sD), stream);
    d_diff = diff_data.data();
  } else {
    d_diff = out;
  }

  // Generate first value of the differenced series
  {
    // Generate the values
    device_buffer<DataT> first_value(allocator, stream);
    first_value.resize(batch_size, stream);
    gpu_gen.uniform(first_value.data(), batch_size, -diff_scale, diff_scale,
                    stream);

    // Copy in the array (TODO: create strided version of random generators?)
    DataT* d_fv = first_value.data();
    thrust::for_each(
      thrust::cuda::par.on(stream), counting, counting + batch_size,
      [=] __device__(int ib) { d_diff[(n_obs - d_sD) * ib] = d_fv[ib]; });
  }

  // Generate noise/residuals
  device_buffer<DataT> residuals(allocator, stream);
  residuals.resize(batch_size * (n_obs - d_sD), stream);
  gpu_gen.normal(residuals.data(), batch_size * (n_obs - d_sD), (DataT)0.0,
                 noise_scale, stream);
  const DataT* d_res = residuals.data();

  // Iterate to generate the differenced series
  thrust::for_each(thrust::cuda::par.on(stream), counting,
                   counting + batch_size, [=] __device__(int ib) {
                     const DataT* b_ar_vec = d_ar_vec + ib * p_sP;
                     const DataT* b_ma_vec = d_ma_vec + ib * q_sQ;
                     const DataT* b_res = d_res + ib * (n_obs - d_sD);
                     DataT* b_diff = d_diff + ib * (n_obs - d_sD);
                     for (int i = 1; i < n_obs - d_sD; i++) {
                       // Noise
                       DataT yi = b_res[i];
                       // AR component
                       for (int ip = 0; ip < p_sP; ip++) {
                         if (i - 1 - ip >= 0)
                           yi += b_ar_vec[ip] * b_diff[i - 1 - ip];
                       }
                       // MA component
                       for (int iq = 0; iq < p_sP; iq++) {
                         if (i - 1 - iq >= 0)
                           yi += b_ma_vec[iq] * b_res[i - 1 - iq];
                       }

                       b_diff[i] = yi;
                     }
                   });

  // Final time series
  if (d_sD || order.k) {
    TimeSeries::finalize_forecast(d_diff, starting_values.data(), n_obs - d_sD,
                                  batch_size, d_sD, d_sD, order.d, order.D,
                                  order.s, stream, order.k, d_mu);
  }

  if (d_sD) {
    // Copy to output
    DataT* d_starting_values = starting_values.data();
    thrust::for_each(thrust::cuda::par.on(stream), counting,
                     counting + batch_size, [=] __device__(int ib) {
                       for (int i = 0; i < d_sD; i++) {
                         out[ib * n_obs + i] = d_starting_values[d_sD * ib + i];
                       }
                       for (int i = 0; i < n_obs - d_sD; i++) {
                         out[ib * n_obs + d_sD + i] =
                           d_diff[(n_obs - d_sD) * ib + i];
                       }
                     });
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace Random
}  // namespace MLCommon
