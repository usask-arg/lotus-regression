from setuptools import setup, find_packages

setup(
    name='LOTUS-regression',
    version='0.1.0',
    packages=find_packages(),
    url='https://arggit.usask.ca/LOTUS/LOTUS-regression',
    license='MIT',
    author='USask ARG',
    author_email='',
    description='Regression analysis code for LOTUS',
    install_requires=['statsmodels', 'numpy', 'pandas', 'scipy']
)
