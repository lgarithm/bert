#!/bin/bash
set -e

ensure_file(){
    local filename=$(basename $1)
    if [ ! -f $filename ]; then
        curl -sLOJ $1
    fi
}

download_bert(){
    mkdir -p $HOME/bert/uncased_L-12_H-768_A-12
    pushd $HOME/bert/uncased_L-12_H-768_A-12
    ensure_file https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip
    unzip uncased_L-12_H-768_A-12.zip
    popd
}

download_squad2(){
    mkdir -p $HOME/dataset/squad2
    pushd $HOME/dataset/squad2
    ensure_file https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
    ensure_file https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
    # ensure_file https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    popd
}


download_bert
download_squad2
