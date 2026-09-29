#!/bin/sh

set -euo pipefail

wget --version &> /dev/null || (echo "wget needs to be installed" && exit 1)

mkdir -p chest-xray-tmp
wget -c 'https://www.kaggle.com/api/v1/datasets/download/muhammadrehan00/chest-xray-dataset' -O ./chest-xray-tmp/chest-xray-dataset.zip
unzip -qq -d ./chest-xray-tmp/full-dataset ./chest-xray-tmp/chest-xray-dataset.zip

mkdir -p data/chest-xray/reference
cp ./chest-xray-tmp/full-dataset/{test,train,val}/*/* ./data/chest-xray/reference/

rm -rf chest-xray-tmp
