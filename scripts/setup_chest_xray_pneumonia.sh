#!/bin/bash

set -euo pipefail

wget --version &> /dev/null || (echo "wget needs to be installed" && exit 1)

mkdir -p chest-xray-pneumonia-tmp
wget -c 'https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/chest-xray-pneumonia' -O ./chest-xray-pneumonia-tmp/chest-xray-pneumonia.zip
unzip -qq -d ./chest-xray-pneumonia-tmp/full-dataset ./chest-xray-pneumonia-tmp/chest-xray-pneumonia.zip

mkdir -p data/chest-xray-pneumonia/reference
cp ./chest-xray-pneumonia-tmp/full-dataset/chest_xray/chest_xray/{test,train,val}/*/* ./data/chest-xray-pneumonia/reference/

rm -rf chest-xray-pneumonia-tmp
