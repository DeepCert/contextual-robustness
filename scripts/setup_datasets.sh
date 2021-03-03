#!/bin/bash

# Usage: ./scripts/setup_datasets.sh

# unzip GTSB dataset
if [ ! -f ./models/gtsb/train.p ]; then
    echo "Unzipping GTSB dataset"
    cd ./data/gtsb && tar -xzf ./test.p.tar.gz && cd -
fi
