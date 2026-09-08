from __future__ import annotations

import numpy as np
import pytest
from statsmodels.regression import linear_model

from LOTUS_regression.regression import mzm_regression


def _correlated_observations(rho):
    rng = np.random.default_rng(42)
    month = np.arange(180)
    predictors = np.column_stack(
        [np.ones(len(month)), month / 120, np.sin(2 * np.pi * month / 12)]
    )
    sigma = 0.2 + 0.1 * (1 + np.cos(2 * np.pi * month / 12))
    residual = rng.standard_normal(len(month))
    for i in range(1, len(month)):
        residual[i] += rho * residual[i - 1]
    observations = predictors @ [2, 0.3, -0.4] + sigma * residual
    observations[[0, 1, 30, 31, 32, 110, 179]] = np.nan
    sigma[51] = 0
    return predictors, observations, sigma


@pytest.mark.parametrize("rho", [-0.6, 0.7])
def test_correlated_gls_matches_covariance_weighted_solution(rho):
    predictors, observations, sigma = _correlated_observations(rho)
    output = mzm_regression(predictors, observations, sigma=sigma)
    result = output["gls_results"]
    model = result.model

    # Independently solve the GLS normal equations for the covariance actually
    # used in the final fit. The returned output covariance is the next iterate.
    precision_x = np.linalg.solve(model.sigma, model.exog)
    information = model.exog.T @ precision_x
    expected_params = np.linalg.solve(information, precision_x.T @ model.endog)
    residual = model.endog - model.exog @ expected_params
    expected_scale = residual @ np.linalg.solve(model.sigma, residual) / result.df_resid
    expected_bse = np.sqrt(np.diag(np.linalg.inv(information)) * expected_scale)

    np.testing.assert_allclose(result.params, expected_params, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result.bse, expected_bse, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        model.cholsigmainv @ model.sigma @ model.cholsigmainv.T,
        np.eye(len(model.endog)),
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize("do_heteroscedasticity", [False, True])
def test_unused_cholesky_triangle_does_not_change_regression(
    monkeypatch, do_heteroscedasticity
):
    predictors, observations, sigma = _correlated_observations(0.7)

    def clean_cholesky(covariance, *, lower):
        assert lower
        return np.linalg.cholesky(covariance)

    def dirty_cholesky(covariance, *, lower):
        factor = clean_cholesky(covariance, lower=lower)
        # Simulate Accelerate's scratch writes even on unaffected CI platforms.
        return factor + np.triu(np.full_like(factor, 0.25), k=1)

    monkeypatch.setattr(linear_model, "cholesky", clean_cholesky)
    expected = mzm_regression(
        predictors,
        observations,
        sigma=sigma.copy(),
        do_heteroscedasticity=do_heteroscedasticity,
    )
    monkeypatch.setattr(linear_model, "cholesky", dirty_cholesky)
    actual = mzm_regression(
        predictors,
        observations,
        sigma=sigma.copy(),
        do_heteroscedasticity=do_heteroscedasticity,
    )

    for attribute in ["params", "bse"]:
        np.testing.assert_allclose(
            getattr(actual["gls_results"], attribute),
            getattr(expected["gls_results"], attribute),
            rtol=1e-10,
            atol=1e-12,
        )
    for key in [
        "fit_values",
        "residual",
        "transformed_residuals",
        "autocorrelation",
        "corrected_weights",
        "covariance",
    ]:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-10, atol=1e-12)
    assert actual["numiter"] == expected["numiter"]
