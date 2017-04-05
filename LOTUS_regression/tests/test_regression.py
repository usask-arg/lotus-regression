import unittest
import pandas as pd
import numpy as np
from scipy.linalg import toeplitz
import statsmodels as sm
import os
from LOTUS_regression.regression import mzm_regression


class TestRegression(unittest.TestCase):
    def setUp(self):
        predictor_file = os.path.join(os.path.dirname(__file__), 'data', 'predictors.csv')

        basis_df = pd.read_csv(predictor_file, parse_dates=True, index_col='time')

        # Only test from 1984 to 2017
        basis_df = basis_df[(basis_df.index > '1984') & (basis_df.index < '2017')]

        X = np.zeros((len(basis_df), 8))

        X[:, 0] = basis_df.qboA.values
        X[:, 1] = basis_df.qboB.values
        X[:, 2] = basis_df.qboC.values
        X[:, 3] = basis_df.enso.values
        X[:, 4] = basis_df.solar.values
        X[:, 5] = basis_df.trop.values
        X[:, 6] = basis_df.linear_post.values
        X[:, 7] = basis_df.linear_pre.values

        self.X = sm.tools.tools.add_constant(X)

    def test_no_error(self):
        test_coeff = [0, 0.04, 0.02, 0, 0.08, 0.02, 0.02, -0.04, 0.01]

        Y = self.X.dot(test_coeff)

        output = mzm_regression(self.X, Y, do_autocorrelation=False, do_heteroscedasticity=False)

        result_coeff = output['gls_results'].params

        self.assertTrue(np.linalg.norm(test_coeff - result_coeff) < 1e-10)

    def test_simple_error_estimates(self):
        test_coeff = [0, 0.04, 0.02, 0, 0.08, 0.02, 0.02, -0.04, 0.01]

        Y = self.X.dot(test_coeff)

        # Do very small random error
        sigma = np.ones_like(Y)*0.01

        covar = np.diag(sigma**2)

        Y_error = Y + np.random.multivariate_normal(np.zeros_like(Y), covar)

        output = mzm_regression(self.X, Y_error, sigma=sigma, do_autocorrelation=False, do_heteroscedasticity=False)

        output_error_estimate = np.diag(np.sqrt(output['gls_results'].cov_params()))

        pinv_x = np.linalg.pinv(self.X)
        actual_error = np.sqrt(np.diag(pinv_x.dot(covar).dot(pinv_x.T)))

        self.assertTrue(np.linalg.norm(output_error_estimate - actual_error) < 1e-3)

    def test_recover_ar1_structure(self):
        test_coeff = [0, 0.04, 0.02, 0, 0.08, 0.02, 0.02, -0.04, 0.01]

        Y = self.X.dot(test_coeff)

        rho = -0.3

        sigma = np.ones_like(Y) * 0.01

        covar = toeplitz(rho**np.arange(0, len(Y))) * (sigma.T.dot(sigma))

        num_runs = 10

        rho_calc = 0
        for i in range(num_runs):
            Y_error = Y + np.random.multivariate_normal(np.zeros_like(Y), covar)
            output = mzm_regression(self.X, Y_error, sigma=sigma, do_autocorrelation=True, do_heteroscedasticity=False)


            rho_calc += output['autocorrelation'] / num_runs

        pass

    def test_wrong_weights(self):
        test_coeff = [0, 0.04, 0.02, 0, 0.08, 0.02, 0.02, -0.04, 0.01]

        Y = self.X.dot(test_coeff)

        # Do very small random error
        sigma = np.ones_like(Y)*0.01

        covar = np.diag(sigma**2)

        Y_error = Y + np.random.multivariate_normal(np.zeros_like(Y), covar)

        in_sigma = sigma + 0.009*np.cos(np.arange(0, len(Y))*2*np.pi/12)

        output = mzm_regression(self.X, Y_error, sigma=in_sigma, do_autocorrelation=False, do_heteroscedasticity=False)

        output_error_estimate = np.diag(np.sqrt(output['gls_results'].cov_params()))

        pinv_x = np.linalg.pinv(self.X)
        actual_error = np.sqrt(np.diag(pinv_x.dot(covar).dot(pinv_x.T)))

        self.assertTrue(np.linalg.norm(output_error_estimate - actual_error) > 1e-4)

    def test_wrong_weights_heteroscedasticity(self):
        test_coeff = [0, 0.04, 0.02, 0, 0.08, 0.02, 0.02, -0.04, 0.01]

        Y = self.X.dot(test_coeff)

        # Do very small random error
        sigma = np.ones_like(Y)*0.01

        covar = np.diag(sigma**2)

        Y_error = Y + np.random.multivariate_normal(np.zeros_like(Y), covar)

        in_sigma = sigma + 0.009*np.cos(np.arange(0, len(Y))*2*np.pi/12)

        output = mzm_regression(self.X, Y_error, sigma=in_sigma, do_autocorrelation=False, do_heteroscedasticity=True)

        output_error_estimate = np.diag(np.sqrt(output['gls_results'].cov_params()))

        pinv_x = np.linalg.pinv(self.X)
        actual_error = np.sqrt(np.diag(pinv_x.dot(covar).dot(pinv_x.T)))

        output = mzm_regression(self.X, Y_error, sigma=in_sigma, do_autocorrelation=False, do_heteroscedasticity=False)

        output_error_estimate_without = np.diag(np.sqrt(output['gls_results'].cov_params()))

        self.assertTrue(np.linalg.norm(output_error_estimate - actual_error) < np.linalg.norm(output_error_estimate_without - actual_error))

    def test_wrong_weight_step_function(self):
        test_coeff = [0, 0.04, 0.02, 0, 0.08, 0.02, 0.02, -0.04, 0.01]

        Y = self.X.dot(test_coeff)

        # Do very small random error
        sigma = np.ones_like(Y)*0.01

        covar = np.diag(sigma**2)

        Y_error = Y + np.random.multivariate_normal(np.zeros_like(Y), covar)

        in_sigma = sigma
        in_sigma[np.floor(len(in_sigma)/2):] *= 4

        output = mzm_regression(self.X, Y_error, sigma=in_sigma, do_autocorrelation=False, do_heteroscedasticity=True)

        output_error_estimate = np.diag(np.sqrt(output['gls_results'].cov_params()))

        pinv_x = np.linalg.pinv(self.X)
        actual_error = np.sqrt(np.diag(pinv_x.dot(covar).dot(pinv_x.T)))

        output = mzm_regression(self.X, Y_error, sigma=in_sigma, do_autocorrelation=False, do_heteroscedasticity=False)

        output_error_estimate_without = np.diag(np.sqrt(output['gls_results'].cov_params()))