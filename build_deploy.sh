PACKAGE_VERSION="$(python -c "import LOTUS_regression; print(LOTUS_regression.__version__)")"

python setup.py install

python setup.py bdist_wheel

mkdir -p "$OUTPUT_DIR/$PACKAGE_VERSION/"
cp dist/*.whl "$OUTPUT_DIR/$PACKAGE_VERSION/"

cd docs
make html

mkdir -p "$OUTPUT_DIR/$PACKAGE_VERSION/docs/"
cp -r build/html/* "$OUTPUT_DIR/$PACKAGE_VERSION/docs/"