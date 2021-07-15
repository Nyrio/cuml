#
# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

###############################################################################
#                             How these tests work                            #
###############################################################################
#
# This test file contains some unit tests and an integration test.
#
# The units tests use the same parameters with cuML and the reference
# implementation to compare strict parity of specific components.
#
# The integration tests compare that, when fitting and forecasting separately,
# out implementation performs better or approximately as good as the reference
# (it mostly serves to test that we don't have any regression)
#
# Note that there are significant differences between our implementation and
# the reference, and perfect parity cannot be expected for integration tests.

from pandas.core.frame import DataFrame
import pytest

from collections import namedtuple
import numpy as np
import os
import warnings

import pandas as pd
from scipy.optimize.optimize import approx_fprime
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

import cudf
import cuml.tsa.arima as arima
from cuml.common.input_utils import input_to_host_array

from cuml.test.utils import stress_param


###############################################################################
#                                  Test data                                  #
###############################################################################

class ARIMAData:
    """Contains a dataset name and associated metadata
    """
    def __init__(self, batch_size, n_obs, n_test, dataset, start, end,
                 tolerance_integration):
        self.batch_size = batch_size
        self.n_obs = n_obs
        self.n_test = n_test
        self.dataset = dataset
        self.start = start
        self.end = end
        self.tolerance_integration = tolerance_integration

        self.n_train = n_obs - n_test


# ARIMA(1,0,1) with intercept
test_101c = ARIMAData(
    batch_size=8,
    n_obs=15,
    n_test=2,
    dataset="long_term_arrivals_by_citizenship",
    start=10,
    end=25,
    tolerance_integration=0.01
)

# ARIMA(0,0,2) with intercept
test_002c = ARIMAData(
    batch_size=7,
    n_obs=20,
    n_test = 2,
    dataset="net_migrations_auckland_by_age",
    start=15,
    end=30,
    tolerance_integration=0.01
)

# ARIMA(0,1,0) with intercept
test_010c = ARIMAData(
    batch_size=4,
    n_obs=17,
    n_test=2,
    dataset="cattle",
    start=10,
    end=25,
    tolerance_integration=0.01
)

# ARIMA(1,1,0)
test_110 = ARIMAData(
    batch_size=1,
    n_obs=137,
    n_test=5,
    dataset="police_recorded_crime",
    start=100,
    end=150,
    tolerance_integration=0.01
)

# ARIMA(0,1,1) with intercept
test_011c = ARIMAData(
    batch_size=16,
    n_obs=28,
    n_test=2,
    dataset="deaths_by_region",
    start=20,
    end=40,
    tolerance_integration=0.05
)

# ARIMA(1,2,1) with intercept
test_121c = ARIMAData(
    batch_size=2,
    n_obs=137,
    n_test=10,
    dataset="population_estimate",
    start=100,
    end=150,
    tolerance_integration=0.01
)

# ARIMA(1,2,1) with intercept (missing observations)
test_121c_missing = ARIMAData(
    batch_size=2,
    n_obs=137,
    n_test=10,
    dataset="population_estimate_missing",
    start=100,
    end=150,
    tolerance_integration=0.01
)

# ARIMA(1,0,1)(1,1,1)_4
test_101_111_4 = ARIMAData(
    batch_size=3,
    n_obs=101,
    n_test=10,
    dataset="alcohol",
    start=80,
    end=110,
    tolerance_integration=0.01
)

# ARIMA(5,1,0)
test_510 = ARIMAData(
    batch_size=3,
    n_obs=101,
    n_test=10,
    dataset="alcohol",
    start=80,
    end=110,
    tolerance_integration=0.02
)

# ARIMA(1,1,1)(2,0,0)_4 with intercept
test_111_200_4c = ARIMAData(
    batch_size=14,
    n_obs=123,
    n_test=10,
    dataset="hourly_earnings_by_industry",
    start=115,
    end=130,
    tolerance_integration=0.01
)

# ARIMA(1,1,1)(2,0,0)_4 with intercept (missing observations)
test_111_200_4c_missing = ARIMAData(
    batch_size=14,
    n_obs=123,
    n_test=10,
    dataset="hourly_earnings_by_industry_missing",
    start=115,
    end=130,
    tolerance_integration=0.01
)

# ARIMA(1,1,2)(0,1,2)_4
test_112_012_4 = ARIMAData(
    batch_size=2,
    n_obs=179,
    n_test=10,
    dataset="passenger_movements",
    start=160,
    end=200,
    tolerance_integration=0.001
)

# ARIMA(1,1,1)(1,1,1)_12
test_111_111_12 = ARIMAData(
    batch_size=12,
    n_obs=279,
    n_test=20,
    dataset="guest_nights_by_region",
    start=260,
    end=290,
    tolerance_integration=0.001
)

# ARIMA(1,1,1)(1,1,1)_12 (missing observations)
test_111_111_12_missing = ARIMAData(
    batch_size=12,
    n_obs=279,
    n_test=20,
    dataset="guest_nights_by_region_missing",
    start=260,
    end=290,
    tolerance_integration=0.001
)

# Dictionary matching a test case to a tuple of model parameters
# (a test case could be used with different models)
# (p, d, q, P, D, Q, s, k) -> ARIMAData
test_data = [
    # ((1, 0, 1, 0, 0, 0, 0, 1), test_101c),
    ((0, 0, 2, 0, 0, 0, 0, 1), test_002c),
    ((0, 1, 0, 0, 0, 0, 0, 1), test_010c),
    ((1, 1, 0, 0, 0, 0, 0, 0), test_110),
    ((0, 1, 1, 0, 0, 0, 0, 1), test_011c),
    ((1, 2, 1, 0, 0, 0, 0, 1), test_121c),
    ((1, 2, 1, 0, 0, 0, 0, 1), test_121c_missing),
    ((1, 0, 1, 1, 1, 1, 4, 0), test_101_111_4),
    ((5, 1, 0, 0, 0, 0, 0, 0), test_510),
    ((1, 1, 1, 2, 0, 0, 4, 1), test_111_200_4c),
    ((1, 1, 1, 2, 0, 0, 4, 1), test_111_200_4c_missing),
    ((1, 1, 2, 0, 1, 2, 4, 0), test_112_012_4),
    stress_param((1, 1, 1, 1, 1, 1, 12, 0), test_111_111_12),
    stress_param((1, 1, 1, 1, 1, 1, 12, 0), test_111_111_12_missing),
]

# Dictionary for lazy-loading of datasets
# (name, dtype) -> (pandas dataframe, cuDF dataframe)
lazy_data = {}

# Dictionary for lazy-evaluation of reference fits
# (p, d, q, P, D, Q, s, k, name, dtype) -> SARIMAXResults
lazy_ref_fit = {}


def extract_order(tup):
    """Extract the order from a tuple of parameters"""
    p, d, q, P, D, Q, s, k = tup
    return (p, d, q), (P, D, Q, s), k


data_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'ts_datasets')


def get_dataset(data, dtype):
    """Load a dataset with a given dtype or return a previously loaded dataset
    """
    key = (data.dataset, np.dtype(dtype).name)
    if key not in lazy_data:
        y = pd.read_csv(
            os.path.join(data_path, "{}.csv".format(data.dataset)),
            usecols=range(1, data.batch_size + 1), dtype=dtype)
        y_train, y_test = train_test_split(y, test_size=data.n_test,
                                           shuffle=False)
        y_train_cudf = cudf.from_pandas(y_train).fillna(np.nan)
        y_test_cudf = cudf.from_pandas(y_test)
        lazy_data[key] = (y_train, y_train_cudf, y_test, y_test_cudf)
    return lazy_data[key]


def get_ref_fit(data, order, seasonal_order, intercept, dtype):
    """Compute a reference fit of a dataset with the given parameters and dtype
    or return a previously computed fit
    """
    y_train, *_ = get_dataset(data, dtype)
    key = order + seasonal_order + \
        (intercept, data.dataset, np.dtype(dtype).name)
    if key not in lazy_ref_fit:
        ref_model = [sm.tsa.SARIMAX(y_train[col], order=order,
                                    seasonal_order=seasonal_order,
                                    trend='c' if intercept else 'n')
                     for col in y_train.columns]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            lazy_ref_fit[key] = [model.fit(disp=0) for model in ref_model]
    return lazy_ref_fit[key]


###############################################################################
#                              Utility functions                              #
###############################################################################

def mase(y_train, y_test, y_fc, s):
    # TODO: fix for missing observations

    y_train_np = input_to_host_array(y_train).array
    y_test_np = input_to_host_array(y_test).array
    y_fc_np = input_to_host_array(y_fc).array

    diff = np.abs(y_train_np[s:] - y_train_np[:-s])
    scale = np.zeros(y_train_np.shape[1])
    for ib in range(y_train_np.shape[1]):
        scale[ib] = diff[~np.isnan(diff)].mean(axis=0)
    scale = diff[~np.isnan(diff[:, ib]), ib].mean()

    error = np.abs(y_fc_np - y_test_np).mean(axis=0)
    return np.mean(error / scale)


###############################################################################
#                                    Tests                                    #
###############################################################################

@pytest.mark.parametrize('key, data', test_data)
@pytest.mark.parametrize('dtype', [np.float64])
def test_integration(key, data, dtype):
    """Full integration test: estimate, fit, forecast
    """
    order, seasonal_order, intercept = extract_order(key)
    s = max(1, seasonal_order[3])

    y_train, y_train_cudf, y_test, _ = get_dataset(data, dtype)

    # Get fit reference model
    ref_fits = get_ref_fit(data, order, seasonal_order, intercept, dtype)

    # Create and fit cuML model
    cuml_model = arima.ARIMA(y_train_cudf,
                             order=order,
                             seasonal_order=seasonal_order,
                             fit_intercept=intercept,
                             output_type='numpy')
    cuml_model.fit()

    # Predict
    y_fc_cuml = cuml_model.forecast(data.n_test)
    y_fc_ref = np.zeros((data.n_test, data.batch_size))
    for i in range(data.batch_size):
        y_fc_ref[:, i] = ref_fits[i].get_prediction(
            data.n_train, data.n_obs - 1).predicted_mean

    # Compare results: MASE must be better or within the tolerance margin
    mase_ref = mase(y_train, y_test, y_fc_ref, s)
    mase_cuml = mase(y_train, y_test, y_fc_cuml, s)
    assert mase_cuml < mase_ref * (1. + data.tolerance_integration)


def _statsmodels_to_cuml(ref_fits, cuml_model, order, seasonal_order,
                         intercept, dtype):
    """Utility function to transfer the parameters from a statsmodels'
    SARIMAXResults object to a cuML ARIMA object.

    .. note:: be cautious with the intercept, it is not always equivalent
        in statsmodels and cuML models (it depends on the order).

    """
    nb = cuml_model.batch_size
    N = cuml_model.complexity
    x = np.zeros(nb * N, dtype=np.float64)

    for ib in range(nb):
        x[ib*N:(ib+1)*N] = ref_fits[ib].params[:N]

    cuml_model.unpack(x)


def _predict_common(key, data, dtype, start, end, num_steps=None, level=None,
                    simple_differencing=True):
    """Utility function used by test_predict and test_forecast to avoid
    code duplication.
    """
    order, seasonal_order, intercept = extract_order(key)

    _, y_train_cudf, *_ = get_dataset(data, dtype)

    # Get fit reference model
    ref_fits = get_ref_fit(data, order, seasonal_order, intercept, dtype)

    # Create cuML model
    cuml_model = arima.ARIMA(y_train_cudf,
                             order=order,
                             seasonal_order=seasonal_order,
                             fit_intercept=intercept,
                             output_type='numpy',
                             simple_differencing=simple_differencing)

    # Feed the parameters to the cuML model
    _statsmodels_to_cuml(ref_fits, cuml_model, order, seasonal_order,
                         intercept, dtype)

    # Predict or forecast
    # Reference (statsmodels)
    ref_preds = np.zeros((end - start, data.batch_size))
    for i in range(data.batch_size):
        ref_preds[:, i] = ref_fits[i].get_prediction(
            start, end - 1).predicted_mean
    if level is not None:
        ref_lower = np.zeros((end - start, data.batch_size))
        ref_upper = np.zeros((end - start, data.batch_size))
        for i in range(data.batch_size):
            temp_pred = ref_fits[i].get_forecast(num_steps)
            ci = temp_pred.summary_frame(alpha=1-level)
            ref_lower[:, i] = ci["mean_ci_lower"].to_numpy()
            ref_upper[:, i] = ci["mean_ci_upper"].to_numpy()
    # cuML
    if num_steps is None:
        cuml_pred = cuml_model.predict(start, end)
    elif level is not None:
        cuml_pred, cuml_lower, cuml_upper = \
            cuml_model.forecast(num_steps, level)
    else:
        cuml_pred = cuml_model.forecast(num_steps)

    # Compare results
    np.testing.assert_allclose(cuml_pred, ref_preds, rtol=0.001, atol=0.01)
    if level is not None:
        np.testing.assert_allclose(
            cuml_lower, ref_lower, rtol=0.005, atol=0.01)
        np.testing.assert_allclose(
            cuml_upper, ref_upper, rtol=0.005, atol=0.01)


@pytest.mark.parametrize('key, data', test_data)
@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('simple_differencing', [True, False])
def test_predict_in(key, data, dtype, simple_differencing):
    """Test in-sample prediction against statsmodels (with the same values
    for the model parameters)
    """
    _predict_common(key, data, dtype, data.n_train // 2, data.n_obs,
                    simple_differencing=simple_differencing)

@pytest.mark.parametrize('key, data', test_data)
@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('simple_differencing', [True, False])
def test_predict_inout(key, data, dtype, simple_differencing):
    """Test in- and ouf-of-sample prediction against statsmodels (with the
    same values for the model parameters)
    """
    _predict_common(key, data, dtype, data.n_train // 2, data.n_train,
                    simple_differencing=simple_differencing)

@pytest.mark.parametrize('key, data', test_data)
@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('simple_differencing', [True, False])
def test_forecast(key, data, dtype, simple_differencing):
    """Test out-of-sample forecasting against statsmodels (with the same
    values for the model parameters)
    """
    _predict_common(key, data, dtype, data.n_train, data.n_obs, data.n_test,
                    simple_differencing=simple_differencing)


@pytest.mark.parametrize('key, data', test_data)
@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('level', [0.5, 0.95])
def test_intervals(key, data, dtype, level):
    """Test forecast confidence intervals against statsmodels (with the same
    values for the model parameters)
    """
    _predict_common(key, data, dtype, data.n_train, data.n_obs, data.n_test,
                    level)


@pytest.mark.parametrize('key, data', test_data)
@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('simple_differencing', [True, False])
def test_loglikelihood(key, data, dtype, simple_differencing):
    """Test loglikelihood against statsmodels (with the same values for the
    model parameters)
    """
    order, seasonal_order, intercept = extract_order(key)

    _, y_train_cudf, *_ = get_dataset(data, dtype)

    # Get fit reference model
    ref_fits = get_ref_fit(data, order, seasonal_order, intercept, dtype)

    # Create cuML model
    cuml_model = arima.ARIMA(y_train_cudf,
                             order=order,
                             seasonal_order=seasonal_order,
                             fit_intercept=intercept,
                             simple_differencing=simple_differencing)

    # Feed the parameters to the cuML model
    _statsmodels_to_cuml(ref_fits, cuml_model, order, seasonal_order,
                         intercept, dtype)

    # Compute loglikelihood
    cuml_llf = cuml_model.llf
    ref_llf = np.array([ref_fit.llf for ref_fit in ref_fits])

    # Compare results
    np.testing.assert_allclose(cuml_llf, ref_llf, rtol=0.01, atol=0.01)


@pytest.mark.parametrize('key, data', test_data)
@pytest.mark.parametrize('dtype', [np.float64])
def test_gradient(key, data, dtype):
    """
    Test batched gradient implementation against scipy non-batched
    gradient.

    .. note:: it doesn't test that the loglikelihood is correct!
    """
    order, seasonal_order, intercept = extract_order(key)
    p, _, q = order
    P, _, Q, _ = seasonal_order
    N = p + P + q + Q + intercept + 1
    h = 1e-8

    _, y_train_cudf, *_ = get_dataset(data, dtype)

    # Create cuML model
    cuml_model = arima.ARIMA(y_train_cudf,
                             order=order,
                             seasonal_order=seasonal_order,
                             fit_intercept=intercept)

    # Get an estimate of the parameters and pack them into a vector
    cuml_model._estimate_x0()
    x = cuml_model.pack()

    # Compute the batched loglikelihood gradient
    batched_grad = cuml_model._loglike_grad(x, h)

    # Iterate over the batch to compute a reference gradient
    scipy_grad = np.zeros(N * data.batch_size)
    for i in range(data.batch_size):
        # Create a model with only the current series
        model_i = arima.ARIMA(y_train_cudf[y_train_cudf.columns[i]],
                              order=order,
                              seasonal_order=seasonal_order,
                              fit_intercept=intercept)

        def f(x):
            return model_i._loglike(x)

        scipy_grad[N * i: N * (i + 1)] = \
            approx_fprime(x[N * i: N * (i + 1)], f, h)

    # Compare
    np.testing.assert_allclose(batched_grad, scipy_grad, rtol=0.001, atol=0.01)


@pytest.mark.parametrize('key, data', test_data)
@pytest.mark.parametrize('dtype', [np.float64])
def test_start_params(key, data, dtype):
    """Test starting parameters against statsmodels
    """
    order, seasonal_order, intercept = extract_order(key)

    y_train, y_train_cudf, *_ = get_dataset(data, dtype)

    # fillna for reference to match cuML initial estimation strategy
    y_train_nona = y_train.fillna(method="ffill").fillna(method="bfill")

    # Create models
    cuml_model = arima.ARIMA(y_train_cudf,
                             order=order,
                             seasonal_order=seasonal_order,
                             fit_intercept=intercept)
    ref_model = [sm.tsa.SARIMAX(y_train_nona[col], order=order,
                                seasonal_order=seasonal_order,
                                trend='c' if intercept else 'n')
                 for col in y_train_nona.columns]

    # Estimate reference starting parameters
    N = cuml_model.complexity
    nb = data.batch_size
    x_ref = np.zeros(N * nb, dtype=dtype)
    for ib in range(nb):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            x_ref[ib*N:(ib+1)*N] = ref_model[ib].start_params[:N]

    # Estimate cuML starting parameters
    cuml_model._estimate_x0()
    x_cuml = cuml_model.pack()

    # Compare results
    np.testing.assert_allclose(x_cuml, x_ref, rtol=0.001, atol=0.01)
