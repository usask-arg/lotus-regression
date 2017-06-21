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


Examples
--------
Some of the examples may require additional packages for loading and handling of data, to obtain all of the packages
necessary to run the examples you may run::

   conda install matplotlib netcdf4 dask hdf5 hdf4 jpeg=8d

.. toctree::
   :maxdepth: 2
   :glob:

   examples/GOZCARDS Trends
   examples/Sage II-OSIRIS-OMPS-LP Trends
   examples/SBUV Trends
   examples/Predictors
   examples/SingleBin
   examples/Changing the Linear Components


API Reference
-------------

.. toctree::
   :maxdepth: 2
   :glob:

   api_reference
