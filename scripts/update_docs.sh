#!/bin/bash

# Generates sphinx apidoc configuration for new modules
# Usage: ./scripts/update_docs.sh

cd build
sphinx-apidoc -o ./source .. 
cd -
