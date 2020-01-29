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

#include <cuml/cuml.hpp>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include "arima/batched_arima.hpp"
#include "cuml/tsa/arima_common.h"
#include "random/rng.h"

#include "benchmark.cuh"

namespace ML {
namespace Bench {
namespace arima {

struct ArimaParams {
  TimeSeriesParams data;
  ARIMAOrder order;
  TimeSeriesScale scale;
};

template <typename D>
class ArimaLoglikelihood : public ArimaFixture<D> {
 public:
  ArimaLoglikelihood(const std::string& name, const ArimaParams& p)
    : ArimaFixture<D>(p.data, p.order, p.scale),
      params(p.data),
      order(p.order) {
    this->SetName(name.c_str());
  }

  // Note: public function because of the __device__ lambda
  void runBenchmark(::benchmark::State& state) override {
    auto& handle = *this->handle;
    auto stream = handle.getStream();
    auto allocator = handle.getDeviceAllocator();
    auto counting = thrust::make_counting_iterator(0);

    // Generate random parameters
    int N = order.complexity();
    D* x = (D*)allocator->allocate(N * params.batch_size, stream);
    MLCommon::Random::Rng gpu_gen(params.seed, MLCommon::Random::GenPhilox);
    gpu_gen.uniform(x, N * params.batch_size, -1.0, 1.0, stream);
    // Set sigma2 parameters to 1.0
    thrust::for_each(thrust::cuda::par.on(stream), counting,
                     counting + params.batch_size,
                     [=] __device__(int bid) { x[bid * (N + 1) - 1] = 1.0; });

    // Create arrays for log-likelihood and residual
    D* loglike = (D*)allocator->allocate(params.batch_size, stream);
    D* res = (D*)allocator->allocate(
      params.batch_size * (params.n_obs - order.lost_in_diff()), stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Benchmark loop
    for (auto _ : state) {
      CudaEventTimer timer(handle, state, true, stream);
      // Evaluate log-likelihood
      batched_loglike(handle, this->data.X, params.batch_size, params.n_obs,
                      order, x, loglike, res, true, false);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Clear memory
    allocator->deallocate(x, order.complexity() * params.batch_size, stream);
    allocator->deallocate(loglike, params.batch_size, stream);
    allocator->deallocate(
      res, params.batch_size * (params.n_obs - order.lost_in_diff()), stream);
  }

 private:
  TimeSeriesParams params;
  ARIMAOrder order;
};

template <typename D>
std::vector<ArimaParams> getInputs() {
  struct std::vector<ArimaParams> out;
  ArimaParams p;
  p.data.seed = 12345ULL;
  p.scale.series = 1.0;
  p.scale.noise = 0.2;
  p.scale.intercept = 1.0;
  std::vector<ARIMAOrder> list_order = {{1, 1, 1, 0, 0, 0, 0, 0},
                                        {1, 1, 1, 1, 1, 1, 4, 0},
                                        {1, 1, 1, 1, 1, 1, 12, 0},
                                        {1, 1, 1, 1, 1, 1, 52, 0}};
  std::vector<int> list_batch_size = {10, 100, 1000, 10000};
  std::vector<int> list_n_obs = {200, 500, 1000};
  for (auto& order : list_order) {
    for (auto& batch_size : list_batch_size) {
      for (auto& n_obs : list_n_obs) {
        p.order = order;
        p.data.batch_size = batch_size;
        p.data.n_obs = n_obs;
        out.push_back(p);
      }
    }
  }
  return out;
}

CUML_BENCH_REGISTER(ArimaParams, ArimaLoglikelihood<double>, "arima",
                    getInputs<double>());

}  // namespace arima
}  // namespace Bench
}  // namespace ML
