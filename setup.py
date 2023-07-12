from setuptools import setup, find_packages
import versioneer

setup(
    name='LOTUS-regression',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    url='http://arg.usask.ca/docs/LOTUS_regression',
    license='MIT',
    author='USask ARG',
    author_email='',
    include_package_data=True,
    description='Regression analysis code for LOTUS',
    long_description_content_type='text/markdown',
    install_requires=['numpy', 'scipy', 'pandas', 'xarray', 'statsmodels', 'requests', 'appdirs']
)
