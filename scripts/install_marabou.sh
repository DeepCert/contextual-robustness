#!/bin/bash

# Usage: ./scripts/install_marabou.sh

if [ ! -d "./marabou" ]; then
    echo "Downloading Marabou"
    git clone https://github.com/NeuralNetworkVerification/Marabou ./marabou
    echo "Building Marabou"
    mkdir ./marabou/build && cd ./marabou/build
    cmake .. -DBUILD_PYTHON=ON -DBUILD_TYPE=Release
    cmake --build . -j4
    cd ../../
fi

echo "Appending marabou to path variables"
MARABOU_PATH="$(pwd)/marabou"
export PYTHONPATH="${PYTHONPATH}:$MARABOU_PATH"
export JUPYTER_PATH="${JUPYTER_PATH}:$MARABOU_PATH"
alias marabou="$MARABOU_PATH/build/Marabou"

cat << EOF
To permantly add Marabou to the path variables, add the following lines to your .bashrc (or .zshrc) file.

MARABOU_PATH="$(pwd)/marabou"
export PYTHONPATH="\$PYTHONPATH:\$MARABOU_PATH"
export JUPYTER_PATH="\$JUPYTER_PATH:\$MARABOU_PATH"
alias marabou="\$MARABOU_PATH/build/Marabou"
EOF
