#!/bin/bash

fpath="$1"
version="$2"

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
cd "$fpath"
if ! test -f "$fpath/H2GKBs.zip"; then
echo "downloading HPO2Gene KnowledgeBase........"
wget https://github.com/WGLab/Phen2Gene/releases/download/1.1.0/H2GKBs.zip
fi
echo "unzipping H2GKBs.zip........"
unzip -q "$fpath/H2GKBs.zip"
rm "$fpath/H2GKBs.zip"


if [[ -z "$version" ]]
then
  if ! test -f "$fpath/PhenosvFile.tar"; then
  echo "downloading PhenoSV files........"
  wget https://www.openbioinformatics.org/PhenoSV/PhenosvFile.tar
  fi
  echo "unzipping PhenosvFile.tar........"
  tar -xvf "$fpath/PhenosvFile.tar"
  rm "$fpath/PhenosvFile.tar"
else
  if ! test -f "$fpath/PhenosvlightFile.tar"; then
  echo "downloading PhenoSV-light files........"
  wget https://www.openbioinformatics.org/PhenoSV/PhenosvlightFile.tar
  fi
  echo "unzipping PhenosvlightFile.tar........"
  tar -xvf "$fpath/PhenosvlightFile.tar"
  rm "$fpath/PhenosvlightFile.tar"
fi


python3 $current_dir/phenosv/setup.py --path "$fpath/data"
