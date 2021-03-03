#!/bin/bash

# Usage: ./scripts/setup_datasets.sh

# unzip GTSB dataset
if [[ ! -f ./data/gtsb/train.p || ! -f ./data/gtsb/test.p ]]; then
    echo "Decompressing GTSB dataset"
    cd ./data/gtsb
    tar -xzf ./train.p.tar.gz
    tar -xzf ./test.p.tar.gz
    cd -
fi
