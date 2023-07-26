#!/bin/bash

svfile=$1
outfolder=$(echo $2| sed 's/\/$//')
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

# loop over the smaller files and call phenosv.py in parallel
echo "start running PhenoSV using $worker workers"
if [ -z "$4" ]
then
	for i in "$svdir/input_"*;do
  		inname=$(basename $i)
  		outname="${inname/input_/output_}"
  		python3 phenosv.py --model 'PhenoSV-light' --sv_file $i --target_folder $outfolder --target_file_name $outname &
	done
else
	for i in "$svdir/input_"*;do
                inname=$(basename $i)
                outname="${inname/input_/output_}"
                python3 phenosv.py --model 'PhenoSV-light' --sv_file $i --HPO $4 --target_folder $outfolder --target_file_name $outname &
        done

fi

wait

# clean and merge
rm $svdir/input_*
awk 'FNR==1 && NR!=1{next;}{print}' $outfolder/output_* > $outfolder/$filename.out.csv
echo "Task completed. Output file is: $outfolder/$filename.out.csv"
rm $outfolder/output_*
