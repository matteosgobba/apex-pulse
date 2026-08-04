#!/bin/sh
set -eu

python -m f1_prediction.runtime initialize
exec "$@"
