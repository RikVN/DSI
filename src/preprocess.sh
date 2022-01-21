#!/bin/bash
# Two arguments:
# $1: folder in which we find DSIs
# $2: iso-code of the language
# ./src/preprocess.sh text.txt en
set -eu -o pipefail
file=$1
lang=$2
maxl="50000"

# Process
echo "Processing $file"
python src/clean_data.py --input_file $file -minl 7 -maxl $maxl  -l $lang -o ${file}.clean
python src/tokenize_and_lowercase.py -i ${file}.clean -o ${file}.clean.tok.lower -l $lang
# Run deduplication on the lowercased/tokenized file and just the cleaned one
cat ${file}.clean.tok.lower | python src/neardup_hash.py > ${file}.clean.tok.lower.dedup
cat ${file}.clean | python src/neardup_hash.py > ${file}.clean.dedup
