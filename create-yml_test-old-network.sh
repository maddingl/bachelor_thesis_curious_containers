#!/bin/bash

# parse arguments and check validity
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --name)                     name="$2";                     shift ;;
		--model_name)               model_name="$2";               shift ;;
        --id_dataset)               id_dataset="$2";               shift ;;
        --ood_dataset)              ood_dataset="$2";              shift ;;
        --test_sample_size)         test_sample_size="$2";         shift ;;
        --individual_normalization) individual_normalization="$2"; shift ;;
        *) echo "Unknown parameter passed: $1. Aborting..."; exit 1 ;;
    esac
    shift
done
if [ -z "$name" ]; then echo "No name provided. Aborting..."; exit 1; fi

file="yml/$name.red.yml"

# copy template to new file
if [ -f "$file" ]; then
  read -r -p "$(pwd)/$file already exists. Override? (y/N) " answer
  case ${answer:0:1} in
    y|Y) ;;
    *) echo "Not overriding. Aborting..."; exit 1 ;;
  esac
fi
cp template.red.yml "$file"

# specify stdout and stderr filenames
sed -i "s/stdout.txt/$name\_stdout.txt/" "$file"
sed -i "s/stderr.txt/$name\_stderr.txt/" "$file"

# insert inputs
for i in individual_normalization \
         test_sample_size \
         ood_dataset \
         id_dataset \
         model_name \
         name; do
  if [ ${!i} ]; then
    something_set=1
    sed -i "/^inputs:/a \ \ $i: \"${!i}\"" "$file"
  fi
done

# remove empty dictionary if any parameter has been set
if [ $something_set ]; then
    sed -i "s/^inputs: {}/inputs:/" "$file"
fi
