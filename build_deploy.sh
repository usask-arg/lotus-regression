PACKAGE_VERSION="$(python -c "import LOTUS_regression; print(LOTUS_regression.__version__)")"

python setup.py install

python setup.py bdist_wheel

[ -z "$CI_COMMIT_TAG" ] && OUTPUT_WHEEL_DIR="$OUTPUT_DIR""dev/"

cp dist/*.whl "$OUTPUT_WHEEL_DIR/"

cd docs
mkdir -p source/images
python make_regression_images.py
make html

[ -z "$CI_COMMIT_TAG" ] && OUTPUT_DOC_DIR="$OUTPUT_DOC_DIR""dev/"

cp -r build/html/* "$OUTPUT_DOC_DIR"