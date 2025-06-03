#!/bin/bash

set -e # exit if err

echo "Building wheel with Poetry..."
poetry build

echo "Reinstalling wheel with pip..."
pip install --force-reinstall dist/*.whl

echo "Done"