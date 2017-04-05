from setuptools import setup, find_packages
import versioneer

setup(
    name='LOTUS-regression',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    url='https://arggit.usask.ca/LOTUS/LOTUS-regression',
    license='MIT',
    author='USask ARG',
    author_email='',
    include_package_data=True,
    description='Regression analysis code for LOTUS',
    install_requires=['statsmodels', 'numpy', 'pandas', 'scipy']
)
