#!/bin/sh

set -euo pipefail

python -c "import numpy; import PIL; print('numpy:', numpy.__version__); print('Pillow:', PIL.__version__)" &> /dev/null || (echo "numpy and pillow need to be installed" && exit 1)
wget --version &> /dev/null || (echo "wget needs to be installed" && exit 1)
7z &> /dev/null || (echo "7zip needs to be installed" && exit 1)

mkdir -p aapm-tmp
wget -c 'https://drive.usercontent.google.com/download?id=1Ov6yyzbnCC_gYNuk6RS6EfvVAoSqKGUC&export=download&confirm=t' -O ./aapm-tmp/dataset.zip
7z x -bd -bb0 -o./aapm-tmp/full-dataset ./aapm-tmp/dataset.zip

mkdir -p ./aapm-tmp/full-dataset/data/fd ./aapm-tmp/full-dataset/data/qd

for dir in fd qd; do
	mv aapm-tmp/full-dataset/data/train/"$dir"/* aapm-tmp/full-dataset/data/"$dir"/

	for file in aapm-tmp/full-dataset/data/test/"$dir"/*; do
		filename=$(basename "$file")
		mv "$file" "aapm-tmp/full-dataset/data/$dir/t$filename"
	done
done

HI=$(tools/convert.py --info aapm-tmp/full-dataset/data/ | awk '$1 == "window" { print $4 }')
tools/convert.py --window 0 $HI ./aapm-tmp/full-dataset/data ./data/aapm

rm -rf aapm-tmp
