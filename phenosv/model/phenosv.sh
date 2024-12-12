#!/bin/bash

worker=1 #set up default value

while [[ $# -gt 0 ]]; do
  case $1 in
    --sv_file)
      svfile="$2"
      shift 2
      ;;
    --target_folder)
      outfolder=$(echo "$2" | sed 's/\/$//')
      shift 2
      ;;
    --workers)
      worker="$2"
      shift 2
      ;;
    --genome)
      genome="$2"
      shift 2
      ;;
    --alpha)
      alpha="$2"
      shift 2
      ;;
    --inference)
      inference_mode="$2"
      shift 2
      ;;
    --model)
      model="$2"
      shift 2
      ;;
    --HPO)
      HPO_terms="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

#svfile and outfolder are required arguments
if [[ -z $svfile || -z $outfolder ]]; then
  echo "Error: --sv_file and --target_folder are required arguments."
  exit 1
fi

svdir=$(dirname $svfile)
filename=$(basename $svfile)
extension="${filename##*.}"
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"


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


for i in "$svdir/input_"*; do
  inname=$(basename "$i")
  outname="${inname/input_/output_}"
  python3 "$SCRIPT_DIR/phenosv.py" \
    --sv_file "$i" \
    --target_folder "$outfolder" \
    --target_file_name "$outname" \
    ${genome:+--genome "$genome"} \
    ${alpha:+--alpha "$alpha"} \
    ${inference_mode:+--inference "$inference_mode"} \
    ${model:+--model "$model"} \
    ${HPO_terms:+--HPO "$HPO_terms"} &
done

wait

# clean and merge
rm $svdir/input_*
awk 'FNR==1 && NR!=1{next;}{print}' $outfolder/output_* > $outfolder/$filename.out.csv
echo "Task completed. Output file is: $outfolder/$filename.out.csv"
rm $outfolder/output_*
