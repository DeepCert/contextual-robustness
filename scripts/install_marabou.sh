#!/bin/bash

# Usage: ./scripts/install_marabou.sh

if [ ! -d "./Marabou" ]; then
    echo "Downloading Marabou"
    git clone https://github.com/NeuralNetworkVerification/Marabou
    echo "Building Marabou"
    mkdir ./Marabou/build && cd ./Marabou/build
    cmake ../
    make -j4 -DBUILD_PYTHON=ON -DBUILD_TYPE=Release
    cd ../../
fi

echo "Appending marabou to path variables"
MARABOU_PATH="$(pwd)/Marabou"
export PYTHONPATH="${PYTHONPATH}:$MARABOU_PATH"
export JUPYTER_PATH="${JUPYTER_PATH}:$MARABOU_PATH"
alias marabou="$MARABOU_PATH/build/Marabou"


