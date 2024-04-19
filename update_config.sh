#!/bin/bash

fpath="$1"

if [[ -z "$fpath" ]]
then
fpath="$PWD/lib"
fi

fpath=$(dirname "$fpath")/$(basename "$fpath")
echo "set up path to store feature files as: $fpath"

mkdir -p "$fpath"
mkdir -p ./lib
echo "$fpath" > ./lib/fpath.config


current_dir="$PWD"

python3 $current_dir/phenosv/setup.py --path "$fpath/data"
