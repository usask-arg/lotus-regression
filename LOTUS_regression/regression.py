import statsmodels.api as sm
import numpy as np
from collections import namedtuple
import xarray as xr
import pandas as pd
from copy import copy


def corrected_ar1_covariance(sigma, gaps, rho):
    """
    Calculates the corrected covariance matrix accounting for AR1 structure, this is the Prais and Winsten covariance
    structure with a modification by Savin and White (1978) to account for gaps in the data.

    Parameters
    ----------
    sigma : np.array
        Length n, square root of the diagonal elements of the uncorrected covariance matrix
    gaps : Iterable of tuples
        Each element of gaps must be a tuple containing a length and index field.  index is the location where the gap
        occurs and length is the number of elements in the gap
    rho : float
        AR1 autocorrelation parameter

    Returns
    -------
    np.ndarray
        Covariance matrix (n, n) which accounts for the AR1 structure

    """
    G = np.zeros((len(sigma), len(sigma)))

    for i in range(len(sigma)):
        if i == 0:
            G[i, i] = np.sqrt(1-rho**2)
        else:
            G[i, i] = 1
            G[i, i-1] = -rho

    if gaps is not None:
        for gap in gaps:
            index = gap.index
            if index >= len(sigma):
                break
            m = gap.length

            g = np.sqrt((1-rho**2) / (1-rho**(2*m+2)))

            G[index, index] = g
            G[index, index-1] = -g*rho**(m+1)

    covar = np.linalg.inv(G.T.dot(G))
    covar *= np.outer(sigma, sigma)

    return covar


def remove_nans_and_find_gaps(X, Y, sigma):
    """
    Preprocesses the data by removing NaN's and in the process finds gaps in the data.

    Parameters
    ----------
    X : np.ndarray
        (nsamples, npredictors)  predictor matrix
    Y : np.array
        (nsamples) observations
    sigma : np.array
        (nsamples) Square root of the diagonal elements of the covariance matrix for Y

    Returns
    -------
    X : np.ndarray

    Y : np.array

    sigma : np.array

    gaps : Iterable tuple

    good_index :


    """
    Gap = namedtuple('Gap', ['length', 'index'])

    good_data = ~np.isnan(Y) & (sigma > 0)
    Y_fixed = np.zeros((np.sum(good_data)))

    gaps = []

    # Y is a little tricky because the index constantly changes, have i go over Y and i_yfix go over Y_fixed
    i = 0
    i_yfix = 0
    while i < len(Y):
        if np.isnan(Y[i]) or (sigma[i] == 0):
            gap_start = i

            while i < len(Y) and (np.isnan(Y[i]) or (sigma[i] == 0)):
                i += 1
            gap_end = i

            gaps.append(Gap(gap_end - gap_start, i_yfix))
        if i_yfix >= len(Y_fixed):
            break
        Y_fixed[i_yfix] = Y[i]
        i += 1
        i_yfix += 1

    return X[good_data, :], Y_fixed, sigma[good_data], gaps, good_data


def _heteroscedasticity_fit_values(num_resid, seasonal_harmonics=(3, 4, 6, 12), extra_predictors=None,
                                   merged_flag=None, treat_merged_periods_differently=True):
    """

    Parameters
    ----------
    num_resid : int
        number of residuals
    seasonal_harmonics : tuple, optional
        Default is (3,4,6,12), these are the harmonics used to create the seasonal predictors.  sins and cosines are used
        as predictor
    extra_predictors : np.ndarray
        (nsamples, x) adds extra predictors to be used in addition to the sins and cosines.

    merged_flag : np.array, optional
        Flag indicating different 'modes' of the merged dataset. For example, this flag is commonly used to describe
        different instaument time periods in a merged dataset.
    """

    month_index = np.arange(0, num_resid)

    if merged_flag is None:
        unique_modes = [0]
        merged_flag = np.zeros(num_resid)
    else:
        unique_modes = np.unique(merged_flag)

    if treat_merged_periods_differently:
        X = np.zeros((num_resid, 2 * len(seasonal_harmonics) * len(unique_modes)))
        for idy, mode in enumerate(unique_modes):
            mask = (merged_flag == mode).astype(int)
            for idx, harmonic in enumerate(seasonal_harmonics):
                X[:, 2*idx + 2 * len(seasonal_harmonics) * idy] = np.cos(2*np.pi*month_index / harmonic) * mask
                X[:, 2*idx + 2 * len(seasonal_harmonics) * idy + 1] = np.sin(2*np.pi*month_index / harmonic) * mask

        if len(unique_modes) > 1:
            # Multiple modes, add constants to allow for varying weights between modes
            X_constants = np.zeros((num_resid, len(unique_modes)))
            for idy, mode in enumerate(unique_modes):
                mask = (merged_flag == mode).astype(int)

                X_constants[:, idy] = mask

            X = np.hstack((X, X_constants))
    else:
        if np.max(unique_modes) == 0:
            num_bits = 1
        else:
            num_bits = int(np.floor(np.log2(np.max(unique_modes)))) + 1
        X = np.zeros((num_resid, 2 * len(seasonal_harmonics) * num_bits))

        for bit in range(num_bits):
            mask = ((merged_flag.astype(int) & 2**bit) > 0).astype(int)
            for idx, harmonic in enumerate(seasonal_harmonics):
                X[:, 2*idx + 2 * len(seasonal_harmonics) * bit] = np.cos(2*np.pi*month_index / harmonic) * mask
                X[:, 2*idx + 2 * len(seasonal_harmonics) * bit + 1] = np.sin(2*np.pi*month_index / harmonic) * mask

        if len(unique_modes) > 1:
            # Multiple modes, add constants to allow for varying weights between modes
            X_constants = np.zeros((num_resid, num_bits))
            for bit in range(num_bits):
                mask = ((merged_flag.astype(int) & 2 ** bit) > 0).astype(int)

                X_constants[:, bit] = mask
            X = np.hstack((X, X_constants))

    if extra_predictors is not None:
        X = np.hstack((X, extra_predictors))

    return X


def heteroscedasticity_correction_factors(residual, fit_functions, log_space=False, damping=0.5):
    """
    Finds the heteroscedasticity correction factors outlined in Damadeo et al. 2014.  This is done by fitting
    log(residual^2) to a set of predictors. Ideally the residuals should be completely random and thus these fit values
    are 0, however that is not usually the case in practice.  The calculated correction factors can vary greatly based
    on the set of predictors used.

    Parameters
    ----------
    residual : np.array
        (nsamples) residuals of the fit.  Note that these are the residuals under the transformed variables of the GLS,
        i.e., including the autocorrelation correction done.

    fit_functions : np.ndarray

    Returns
    -------
    Multiplicative correction factors to be applied to the square root of the diagonal elements of the covariance
    matrix.

    """
    Y = residual**2
    if log_space:
        Y = np.log(Y)
    model = sm.OLS(Y, fit_functions)

    results = model.fit()

    f = results.fittedvalues
    if log_space:
        f = np.exp(f)

    correction_factors = (np.sqrt(np.abs(f)) - 1)*damping + 1

    # Sometimes we can get values really close to 0, so dont let the weights change by more than 2 orders of magnitude
    correction_factors[correction_factors <= 1e-2] = 1e-2
    correction_factors[correction_factors >= 1e2] = 1e2

    return correction_factors


def mzm_regression(X, Y, sigma=None, tolerance=1e-5, max_iter=50, do_autocorrelation=True, do_heteroscedasticity=False,
                   verbose_output=False, extra_heteroscedasticity=None, heteroscedasticity_merged_flag=None,
                   treat_merged_periods_differently=True, seasonal_harmonics=(3, 4, 6, 12)):
    """


    Parameters
    ----------
    X : np.ndarray
        (nsamples, npredictors) Array of predictor basis functions
    Y : np.array
        (nsamples) Observations
    sigma : np.array
        (nsamples) Square root of the diagonal elements of the covariance matrix for Y
    tolerance : float, optional
        Iterations stop when the relative difference in the AR1 coefficient is less than this threshold.  Default 1e-5
    max_iter : int, optional
        Maximum number of iterations to perform.  Default 50
    do_autocorrelation : bool, optional
        If true, do the AR1 autocorrelation correction on the covariance matrix.  Default True.
    do_heteroscedasticity : bool, optional
        If True, do the heteroscedasticity correction on the covariance matrix.  Default True.
    verbose_output :
    extra_heteroscedasticity : np.ndarray
        (nsamples, nextrapredictors) Extra predictor functions to use in the heteroscedasticity correction.

    Returns
    -------

    """
    # If we have no weights then still do a weighted regression but use sigma=1
    if sigma is None:
        sigma = np.ones_like(Y, dtype=float)

    # Set up some output variables, we want the output to be the same shape as the input but we have to preprocess the
    # input and remove NaN's
    residuals = np.ones_like(Y)*np.nan
    fit_values = np.ones_like(Y)*np.nan
    transformed_residuals_out = np.ones_like(Y)*np.nan
    sigma_out = np.ones_like(Y)*np.nan

    # Do some preprocessing
    X, Y, sigma, gaps, good_index = remove_nans_and_find_gaps(X, Y, sigma)
    if extra_heteroscedasticity is not None:
        extra_heteroscedasticity = extra_heteroscedasticity[good_index]

    if heteroscedasticity_merged_flag is not None:
        heteroscedasticity_merged_flag = heteroscedasticity_merged_flag[good_index]

    if do_heteroscedasticity:
        heteroscedasticity_X = _heteroscedasticity_fit_values(len(Y), extra_predictors=extra_heteroscedasticity, merged_flag=heteroscedasticity_merged_flag,
                                                              treat_merged_periods_differently=treat_merged_periods_differently,
                                                              seasonal_harmonics=seasonal_harmonics)

    # Initial covariance matrix for the GLS with diagonal structure
    covar = np.diag(sigma**2)

    # Set up some variables for convergence testing, funny value but |rho| has to be less than one so this should be
    # okay
    rho_prior = -10000

    # Output
    output = dict()

    # Main loop
    for i in range(max_iter):
        # GLS Model
        model = sm.GLS(Y, X, covar)
        results = model.fit()

        # Find the autocorrelation coefficient, if we are not doing autocorrelation set rho to 0 so that later our
        # variable transformations are still valid
        if do_autocorrelation:
            rho, _ = sm.regression.yule_walker(results.resid, order=1)
        else:
            rho = 0

        # Find the residuals in the "transformed" GLS units.
        transformed_residuals = np.dot(results.resid, model.cholsigmainv)

        if do_heteroscedasticity and i > 0:
            correction_factors = heteroscedasticity_correction_factors(transformed_residuals, heteroscedasticity_X,
                                                                       log_space=True)
            sigma *= correction_factors
            if not do_autocorrelation:
                # If we arent doing autocorrelation we have to reset the covariance matrix, if we are this will be done
                # in the next step
                covar = np.diag(sigma**2)

        if do_autocorrelation:
            covar = corrected_ar1_covariance(sigma, gaps, rho)

        if not do_autocorrelation and not do_heteroscedasticity:
            # There is nothing here to iterate,
            break
        # If converged we can stop
        if np.abs((rho - rho_prior)) < tolerance and i > 0:
            if not do_heteroscedasticity:
                break
            else:
                if np.linalg.norm(correction_factors - 1) < tolerance:
                    break
        else:
            rho_prior = rho

    residuals[good_index] = results.resid
    fit_values[good_index] = results.fittedvalues

    output['gls_results'] = results
    output['residual'] = residuals

    if do_autocorrelation:
        transformed_residuals_out[good_index] = transformed_residuals
        output['transformed_residuals'] = transformed_residuals_out

    output['autocorrelation'] = rho

    sigma_out[good_index] = sigma
    output['corrected_weights'] = sigma_out
    output['numiter'] = i
    output['fit_values'] = fit_values
    output['covariance'] = covar

    return output


def regress_all_bins(predictors, mzm_data, time_field='time', debug=False, **kwargs):
    mzm_data = mzm_data.rename({time_field, 'time'})

    mzm_data = mzm_data.reindex(time=pd.date_range(mzm_data.time.values[0], mzm_data.time.values[-1], freq=pd.DateOffset(months=1)),
                                tolerance=pd.Timedelta(days=1), method='nearest')

    min_time = mzm_data.time.values[0]
    max_time = mzm_data.time.values[-1]

    min_time = pd.to_datetime(min_time)

    if min_time.day != 1:
        min_time -= pd.DateOffset(months=1)

    predictors = predictors[(predictors.index >= min_time) & (predictors.index <= max_time)]
    pred_list = list(predictors.columns.values)

    # (nsamples, npredictors) matrix
    X = predictors.values

    coords = []
    for c in mzm_data.coords.dims:
        if c != 'time':
            coords.append(c)

    assert(len(coords) <= 2)

    c_list = {c: mzm_data[c].values for c in coords}

    ret = xr.Dataset(coords=c_list)

    sized_nans = np.ones([len(mzm_data[c].values) for c in coords]) * np.nan

    for pred in pred_list:
        ret[pred] = (coords, copy(sized_nans))
        ret[pred + "_std"] = (coords, copy(sized_nans))

    for id_x, x in enumerate(mzm_data[coords[0]].values):
        for id_y, y in enumerate(mzm_data[coords[1]].values):

            sliced_data = mzm_data.loc[{coords[0]: x, coords[1]: y}]
            Y = sliced_data.values
            try:
                output = mzm_regression(X, Y, **kwargs)
                std_error = np.sqrt(np.diag(output['gls_results'].cov_params()))

                for idx, col in enumerate(pred_list):
                    ret[col][id_x, id_y] = output['gls_results'].params[idx]
                    ret[col + '_std'][id_x, id_y] = std_error[idx]

            except Exception as e:
                if debug:
                    print(e)

    return ret