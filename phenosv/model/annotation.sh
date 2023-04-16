#!/bin/bash

svfile=$1
outfolder=$2
worker=$3


svdir=$(dirname $svfile)
filename=$(basename $svfile)
extension="${filename##*.}"

line=$(wc -l $svfile | awk '{print $1}')
v=$(($line/$worker))

echo "read $line lines from $svfile"

# split the input file into n smaller files and rename
split -l $v $svfile "$svdir/input_"

for i in "$svdir/input_"*;do
  mv -- "$i" "${i}.${extension}"
done

# loop over the smaller files and call annotation.py in parallel
echo "start running PhenoSV using $worker workers"

for i in "$svdir/input_"*;do
  	python3 annotation.py --sv_file $i --target $outfolder &
done


wait

# clean and merge
rm $svdir/input_*
