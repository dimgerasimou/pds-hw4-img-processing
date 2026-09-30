#!/bin/bash

set -euo pipefail

python3 -c "import numpy; import PIL; print('numpy:', numpy.__version__); print('Pillow:', PIL.__version__)" &> /dev/null || (echo "numpy and pillow need to be installed" && exit 1)
wget --version &> /dev/null || (echo "wget needs to be installed" && exit 1)

mkdir -p 2detect-tmp
wget -c "https://zenodo.org/records/8017583/files/2DeteCT_slices1-1000_RecSeg.zip?download=1" -O ./2detect-tmp/2detect.zip
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -q ./2detect-tmp/2detect.zip '*/mode1/reconstruction.tif' '*/mode2/reconstruction.tif' -d ./2detect-tmp/2detect

HI=$(tools/convert.py --reference-file mode2/reconstruction.tif --input-file mode1/reconstruction.tif --info 2detect-tmp/2detect | awk '$1 == "window" { print $4 }')
tools/convert.py --window 0 $HI --reference-file mode2/reconstruction.tif --input-file mode1/reconstruction.tif 2detect-tmp/2detect data/2detect

rm -rf 2detect-tmp
