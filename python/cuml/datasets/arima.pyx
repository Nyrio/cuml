#
# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import cuml
import numpy as np

from cuml.common.handle cimport cumlHandle
from cuml.utils import get_dev_array_ptr, zeros

from libc.stdint cimport uint64_t, uintptr_t

from random import randint

cdef extern from "cuml/tsa/arima_common.h" namespace "ML":
    ctypedef struct ARIMAOrder:
        int p  # Basic order
        int d
        int q
        int P  # Seasonal order
        int D
        int Q
        int s  # Seasonal period
        int k  # Fit intercept?

cdef extern from "cuml/datasets/make_arima.hpp" namespace "ML":
    void cpp_make_arima "ML::Datasets::make_arima" (
        const cumlHandle& handle,
        float* out,
        int batch_size,
        int n_obs,
        ARIMAOrder order,
        float scale,
        float noise_scale,
        float diff_scale,
        float* d_mu,
        float* d_ar,
        float* d_ma,
        float* d_sar,
        float* d_sma,
        uint64_t seed
    )

    void cpp_make_arima "ML::Datasets::make_arima" (
        const cumlHandle& handle,
        double* out,
        int batch_size,
        int n_obs,
        ARIMAOrder order,
        double scale,
        double noise_scale,
        double diff_scale,
        double* d_mu,
        double* d_ar,
        double* d_ma,
        double* d_sar,
        double* d_sma,
        uint64_t seed
    )

inp_to_dtype = {
    'single': np.float32,
    'float': np.float32,
    'double': np.float64,
    np.float32: np.float32,
    np.float64: np.float64
}


def make_arima(batch_size=1000, n_obs=100, order=(1, 1, 1),
               seasonal_order=(0, 0, 0, 0), fit_intercept=False,
               scale = 1.0, noise_scale = 0.1, diff_scale = 0.3,
               coef=False, random_state=None,
               dtype='double', handle=None):
    """TODO: docs
    """

    cdef ARIMAOrder cpp_order
    cpp_order.p, cpp_order.d, cpp_order.q = order
    cpp_order.P, cpp_order.D, cpp_order.Q, cpp_order.s = seasonal_order
    cpp_order.k = <int>fit_intercept
    
    if dtype not in ['single', 'float', 'double', np.float32, np.float64]:
        raise TypeError("dtype must be either 'float' or 'double'")
    else:
        dtype = inp_to_dtype[dtype]

    handle = cuml.common.handle.Handle() if handle is None else handle
    cdef cumlHandle* handle_ = <cumlHandle*><size_t>handle.getHandle()

    out = zeros((n_obs, batch_size), dtype=dtype, order='F')
    cdef uintptr_t out_ptr = get_dev_array_ptr(out)

    if random_state is None:
        random_state = randint(0, 1e18)

    # Create mu, ar and ma arrays
    cdef uintptr_t d_mu_ptr = <uintptr_t> NULL
    cdef uintptr_t d_ar_ptr = <uintptr_t> NULL
    cdef uintptr_t d_ma_ptr = <uintptr_t> NULL
    cdef uintptr_t d_sar_ptr = <uintptr_t> NULL
    cdef uintptr_t d_sma_ptr = <uintptr_t> NULL
    if coef and cpp_order.k:
        d_mu = zeros(batch_size, dtype=dtype)
        d_mu_ptr = get_dev_array_ptr(d_mu)
    if coef and cpp_order.p:
        d_ar = zeros((cpp_order.p, batch_size), dtype=dtype, order='F')
        d_ar_ptr = get_dev_array_ptr(d_ar)
    if coef and cpp_order.q:
        d_ma = zeros((cpp_order.q, batch_size), dtype=dtype, order='F')
        d_ma_ptr = get_dev_array_ptr(d_ma)
    if coef and cpp_order.P:
        d_sar = zeros((cpp_order.P, batch_size), dtype=dtype, order='F')
        d_sar_ptr = get_dev_array_ptr(d_sar)
    if coef and cpp_order.Q:
        d_sma = zeros((cpp_order.Q, batch_size), dtype=dtype, order='F')
        d_sma_ptr = get_dev_array_ptr(d_sma)

    if dtype == np.float32:
        cpp_make_arima(handle_[0], <float*> out_ptr, <int> batch_size,
                       <int> n_obs, cpp_order, <float> scale,
                       <float> noise_scale, <float> diff_scale,
                       <float*> d_mu_ptr, <float*> d_ar_ptr,
                       <float*> d_ma_ptr, <float*> d_sar_ptr,
                       <float*> d_sma_ptr, <uint64_t> random_state)

    else:
        cpp_make_arima(handle_[0], <double*> out_ptr, <int> batch_size,
                       <int> n_obs, cpp_order, <double> scale,
                       <double> noise_scale, <double> diff_scale,
                       <double*> d_mu_ptr, <double*> d_ar_ptr,
                       <double*> d_ma_ptr, <double*> d_sar_ptr,
                       <double*> d_sma_ptr, <uint64_t> random_state)

    if coef:
        params = dict()
        if cpp_order.k:
            params["mu"] = d_mu.copy_to_host()
        if cpp_order.p:
            params["ar"] = d_ar.copy_to_host()
        if cpp_order.q:
            params["ma"] = d_ma.copy_to_host()
        if cpp_order.P:
            params["sar"] = d_sar.copy_to_host()
        if cpp_order.Q:
            params["sma"] = d_sma.copy_to_host()
        return out, params
    else:
        return out
