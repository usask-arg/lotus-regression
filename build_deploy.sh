PACKAGE_VERSION="$(python -c "import LOTUS_regression; print(LOTUS_regression.__version__)")"

conda env create --file docs/environment.yml > /dev/null 2>&1
source activate doc_env
python setup.py install

python setup.py bdist_wheel

cp dist/*.whl "$OUTPUT_DIR/$PACKAGE_VERSION/

mkdir -p "$OUTPUT_DIR/$PACKAGE_VERSION/

cd docs
make html

mkdir -p "$OUTPUT_DIR/$PACKAGE_VERSION/docs/"
cp -r build/html/* "$OUTPUT_DIR/$PACKAGE_VERSION/docs/"