.. LOTUS Regression documentation master file, created by
   sphinx-quickstart on Wed Apr  5 11:07:39 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

LOTUS Regression
================

Installation
------------

Requirements
^^^^^^^^^^^^
The code is tested on Python versions `2.7, 3.4, 3.5, 3.6` and should work on any of them,
however we recommend using the `Anaconda python distribution <https://www.continuum.io/downloads>`_ which by default comes with Python
version `3.6`.  Some of the example code may also be incompatible with Python `2.7`

A minimal set of requirements are the Python packages `numpy, scipy, pandas, statsmodels, xarray, requests, appdirs`
which should be automatically be installed upon installing the package, if you are using anaconda you can verify/install
that these packages are available by running::

   conda install numpy scipy pandas xarray statsmodels requests appdirs

How To Install
^^^^^^^^^^^^^^
The code can be installed by running::

   pip install LOTUS_regression -f https://arg.usask.ca/wheels/


What's New in Version 0.5
-------------------------
The largest change has been the addition of preconstructed finalized baseline predictor files.  These three files can
be loaded in with::

   from LOTUS_regression.predictors import load_data

   predictors_pwlt = load_data('pred_baseline_pwlt.csv')
   predictors_ilt = load_data('pred_baseline_ilt.csv')
   predictors_eesc = load_data('pred_baseline_eesc.csv')

All of the satellite dataset examples have been modified to use the `pred_baseline_pwlt.csv` file.  A detailed
description of the predictors contained in the three files can be found on the new technical note page, which in
addition also describes the regression algorithm.

.. toctree::
   :maxdepth: 2
   :glob:

   Technical Note

Other Changes
^^^^^^^^^^^^^
   * Predictors module has been updated to include ILT terms
   * GISS AOD loader has been modified to extend the AOD to the current time based on
     the last non-zero value
   * If linear/EESC terms are included in the first stage of the two-step ILT they are not subtracted when calculating
     the residuals for the second step (two-step ILT is mostly deprecated in favor of a one step ILT but is still available
     for testing purposes)


Examples
--------
Some of the examples may require additional packages for loading and handling of data, to obtain all of the packages
necessary to run the examples you may run::

   conda install matplotlib netcdf4 dask hdf5 hdf4 jpeg=8d


Satellite Datasets
^^^^^^^^^^^^^^^^^^
Here the piecewise linear trends are calculated for three satellite datasets.  Both the GOZCARDS and SBUV datasets
are not deseasonalized, so seasonal components are added to the predictors prior to fitting.

.. toctree::
   :maxdepth: 2
   :glob:

   examples/GOZCARDS Trends
   examples/Sage II-OSIRIS-OMPS-LP Trends
   examples/SBUV Trends

Other Functionality
^^^^^^^^^^^^^^^^^^^
Here are several examples which go through some advanced functionality of the regression package.
The predictors example shows how different predictors can be created for the regression, the SingleBin example
runs the regression for a single bin without using any of the higher level wrappers, and the Linear Components example
shows how the linear components of the fit can be changed.

.. toctree::
   :maxdepth: 2
   :glob:

   examples/Predictors
   examples/SingleBin
   examples/Changing the Linear Components


API Reference
-------------

.. toctree::
   :maxdepth: 2
   :glob:

   api_reference
