#!/bin/bash

# Usage: ./scripts/setup_datasets.sh

# unzip GTSRB dataset
if [[ ! -f ./data/gtsrb/train.p || ! -f ./data/gtsrb/test.p ]]; then
    echo "Decompressing GTSRB dataset"
    cd ./data/gtsrb
    tar -xzf ./train.p.tar.gz
    tar -xzf ./test.p.tar.gz
    cd -
fi
