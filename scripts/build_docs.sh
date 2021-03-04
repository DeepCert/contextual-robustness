#!/bin/bash

# Builds HTML documentation
# Usage: ./scripts/build_docs.sh

cd docs
make clean && make html
cd -
