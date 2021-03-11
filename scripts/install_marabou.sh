#!/bin/bash

# Usage: ./scripts/install_marabou.sh [-h] [-g GUROBI_PATH]
function usage {
    echo "Usage:"
    echo "./$(basename $0) [-h] [-g GUROBI_PATH]"
    echo "-h : show help and exit"
    echo "-g GUROBI_PATH : builds marabou with Gurobi optimizer installed at GUROBI_PATH"
}

GUROBI_PATH=""

while getopts ':hg:' arg; do
  case ${arg} in
    h)
      usage
      exit 0
      ;;
    g)
      GUROBI_PATH="$OPTARG"
      ;;
    :)
      echo "$0: Must supply an argument to -$OPTARG." >&2
      exit 1
      ;;
    ?)
      echo "Invalid option: -${OPTARG}."
      exit 2
      ;;
  esac
done

MARABOU_CONFIG="-DBUILD_PYTHON=ON"

if [ -n "$GUROBI_PATH" ]; then
    MARABOU_CONFIG="$MARABOU_CONFIG -DENABLE_GUROBI=ON -DGUROBI_DIR=$GUROBI_PATH"
fi

if [ ! -d "./marabou" ]; then
    echo "Downloading Marabou"
    git clone https://github.com/NeuralNetworkVerification/Marabou ./marabou
    echo "Building Marabou"
    mkdir ./marabou/build && cd ./marabou/build
    cmake .. $MARABOU_CONFIG
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
