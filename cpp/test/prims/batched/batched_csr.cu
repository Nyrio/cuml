/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>
#include <cuml/common/cuml_allocator.hpp>
#include <cuml/cuml.hpp>
#include <random>
#include <vector>

#include "batched_matrix.h"
#include "linalg/batched/batched_matrix.h"
#include "sparse/batched/csr.h"
#include "test_utils.h"

namespace MLCommon {
namespace Sparse {
namespace Batched {

enum BatchedCSROperation { SpMV_op, SpMM_op };

template <typename T>
struct BatchedCSRInputs {
  BatchedCSROperation operation;
  int batch_size;
  int m;  // Dimensions of A
  int n;
  int nnz;  // Number of non-zero elements in A
  int p;    // Dimensions of B or x
  int q;
  T tolerance;
};

template <typename T>
class BatchedCSRTest : public ::testing::TestWithParam<BatchedCSRInputs<T>> {
 protected:
  void SetUp() override {
    using std::vector;
    params = ::testing::TestWithParam<BatchedCSRInputs<T>>::GetParam();

    // Check if the dimensions are valid and compute the output dimensions
    int m_r, n_r;
    switch (params.operation) {
      case SpMV_op:
        ASSERT_TRUE(params.n == params.p);
        ASSERT_TRUE(params.q == 1);
        m_r = params.m;
        n_r = 1;
        break;
      case SpMM_op:
        ASSERT_TRUE(params.n == params.p);
        m_r = params.m;
        n_r = params.q;
        break;
    }

    // Create test matrices/vectors
    std::vector<T> A;
    std::vector<T> Bx;
    A.resize(params.batch_size * params.m * params.n, (T)0.0);
    Bx.resize(params.batch_size * params.p * params.q);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> idis(0, params.m * params.n - 1);
    std::uniform_real_distribution<T> udis(-1.0, 3.0);

    // Generate a random sparse matrix (with dense representation)
    std::vector<bool> mask = std::vector<bool>(params.m * params.n, false);
    for (int idx = 0; idx < params.nnz; idx++) {
      int k;
      do {
        k = idis(gen);
      } while (mask[k]);
      mask[k] = true;
      int i = k % params.m;
      int j = k / params.m;
      for (int bid = 0; bid < params.batch_size; bid++) {
        A[bid * params.m * params.n + j * params.m + i] = udis(gen);
      }
    }

    // Generate a random dense matrix/vector
    for (int i = 0; i < Bx.size(); i++) Bx[i] = udis(gen);

    // Create handles, stream, allocator
    CUBLAS_CHECK(cublasCreate(&handle));
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto allocator = std::make_shared<MLCommon::defaultDeviceAllocator>();

    // Created batched dense matrices
    LinAlg::Batched::BatchedMatrix<T> AbM(params.m, params.n, params.batch_size,
                                          handle, allocator, stream);
    LinAlg::Batched::BatchedMatrix<T> BxbM(
      params.p, params.q, params.batch_size, handle, allocator, stream);

    // Copy the data to the device
    updateDevice(AbM.raw_data(), A.data(), A.size(), stream);
    updateDevice(BxbM.raw_data(), Bx.data(), Bx.size(), stream);

    // Create sparse matrix A from the dense A and the mask
    BatchedCSR<T> AbS(AbM, mask);

    // Create matrix that will hold the results
    res_bM = new LinAlg::Batched::BatchedMatrix<T>(m_r, n_r, params.batch_size,
                                                   handle, allocator, stream);

    // Compute the tested results
    switch (params.operation) {
      case SpMV_op:
        b_spmv((T)1.0, AbS, BxbM, (T)0.0, *res_bM);
        break;
      case SpMM_op:
        b_spmm((T)1.0, AbS, BxbM, (T)0.0, *res_bM);
        break;
    }

    // Compute the expected results
    res_h.resize(params.batch_size * m_r * n_r);
    switch (params.operation) {
      case SpMV_op:
        for (int bid = 0; bid < params.batch_size; bid++) {
          LinAlg::Naive::naiveMatMul(
            res_h.data() + bid * m_r, A.data() + bid * params.m * params.n,
            Bx.data() + bid * params.p, params.m, params.n, 1);
        }
        break;
      case SpMM_op:
        for (int bid = 0; bid < params.batch_size; bid++) {
          LinAlg::Naive::naiveMatMul(res_h.data() + bid * m_r * n_r,
                                     A.data() + bid * params.m * params.n,
                                     Bx.data() + bid * params.p * params.q,
                                     params.m, params.n, params.q);
        }
        break;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void TearDown() override {
    delete res_bM;
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

 protected:
  BatchedCSRInputs<T> params;
  LinAlg::Batched::BatchedMatrix<T> *res_bM;
  std::vector<T> res_h;
  cublasHandle_t handle;
  cudaStream_t stream;
};

// Test parameters (op, batch_size, m, n, nnz, p, q, tolerance)
const std::vector<BatchedCSRInputs<double>> inputsd = {
    {SpMV_op, 1, 90, 150, 440, 150, 1, 1e-6},
    {SpMV_op, 5, 13, 12, 75, 12, 1, 1e-6},
    {SpMV_op, 15, 8, 4, 6, 4, 1, 1e-6},
    {SpMV_op, 33, 7, 7, 23, 7, 1, 1e-6},
    {SpMM_op, 1, 20, 15, 55, 15, 30, 1e-6},
    {SpMM_op, 9, 10, 9, 31, 9, 11, 1e-6},
    {SpMM_op, 20, 7, 12, 11, 12, 13, 1e-6}
};

// Test parameters (op, batch_size, m, n, nnz, p, q, tolerance)
const std::vector<BatchedCSRInputs<float>> inputsf = {
  {SpMV_op, 1, 90, 150, 440, 150, 1, 1e-2},
  {SpMV_op, 5, 13, 12, 75, 12, 1, 1e-2},
  {SpMV_op, 15, 8, 4, 6, 4, 1, 1e-2},
  {SpMV_op, 33, 7, 7, 23, 7, 1, 1e-2},
  {SpMM_op, 1, 20, 15, 55, 15, 30, 1e-2},
  {SpMM_op, 9, 10, 9, 31, 9, 11, 1e-2},
  {SpMM_op, 20, 7, 12, 11, 12, 13, 1e-2}
};

using BatchedCSRTestD = BatchedCSRTest<double>;
using BatchedCSRTestF = BatchedCSRTest<float>;
TEST_P(BatchedCSRTestD, Result) {
  ASSERT_TRUE(devArrMatchHost(res_h.data(), res_bM->raw_data(), res_h.size(),
                              CompareApprox<double>(params.tolerance), stream));
}
TEST_P(BatchedCSRTestF, Result) {
  ASSERT_TRUE(devArrMatchHost(res_h.data(), res_bM->raw_data(), res_h.size(),
                              CompareApprox<float>(params.tolerance), stream));
}

INSTANTIATE_TEST_CASE_P(BatchedCSRTests, BatchedCSRTestD,
                        ::testing::ValuesIn(inputsd));
INSTANTIATE_TEST_CASE_P(BatchedCSRTests, BatchedCSRTestF,
                        ::testing::ValuesIn(inputsf));

}  // namespace Batched
}  // namespace Sparse
}  // namespace MLCommon