#!/bin/sh
set -e

if [ ! -d KungFu ]; then
    git clone https://github.com/lsds/KungFu.git
fi

export GOBIN=$PWD/bin

cd KungFu
git checkout osdi20-artifact
pip3 install --no-index -U .
go install -v ./...
