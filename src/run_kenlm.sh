#!/bin/bash
set -eu -o pipefail
# In this script I assume we run from the main Github folder
# and that the arguments are not full paths, but local (e.g. exp/ data/ etc)
# We add the current directory to the paths so we can cd to the kenLM folder
# and still use the correct filenames
cur_dir=$(pwd)

# First argument is the experiment folder we setup
exp_dir="${cur_dir}/$1"

# Sanity check: if the folder already exists, return an error, don't want to overwrite
if [ -d $exp_dir ]; then
    echo "Directory $1 already exists, quitting..." ; exit -1
fi

mkdir -p $exp_dir
mkdir -p $exp_dir/data $exp_dir/eval $exp_dir/log

# Location of folders and files
kenlm_home="tools/kenlm/build/"
data_fol="${cur_dir}/data/crawl_19_10_2021/"
ppl_script="src/perplexity_table.py"


# Experimental settings
ngram="3"
train_file="train.clean.tok.lower.dedup"
test_file="test.clean.tok.lower.dedup"
lang="en"

# Set dsis we are working with - e-procurement and ec-europa are excluded
dsis="cybersecurity e-health online-dispute-resolution safer-internet e-justice europeana open-data-portal"

# Train a kenLM language model and evaluate on all other files
for dsi in $dsis ; do
	# Select file
	orig_file="${data_fol}/${dsi}/$lang/${train_file}"
	# Create exp folder for this dsi
	cur_fol="${exp_dir}/data/${dsi}/"
	mkdir -p $cur_fol
	# Train KenLM
	cd $kenlm_home
	./bin/lmplz -o $ngram -S 100G -T /tmp < $orig_file > ${cur_fol}/model.arpa
	# Make model a binary
	./bin/build_binary ${cur_fol}/model.arpa ${cur_fol}/model.binary
	# Evaluate on models
	for eval in $dsis ; do
		eval_file="${data_fol}/${eval}/$lang/${test_file}"
		./bin/query ${cur_fol}model.binary < $eval_file > ${cur_fol}/${eval}.eval
	done
	# Back to original dir
	cd $cur_dir
done

# Print nice perplexity table and save to file, for both including and excluding oov tokens
python $ppl_script --input_folder ${exp_dir}/data/ --oov including > ${exp_dir}/eval/eval.incl
python $ppl_script --input_folder ${exp_dir}/data/ --oov excluding > ${exp_dir}/eval/eval.excl

# Save all experimental values from this script to a file
# This way we can always go back and check how we obtained the results
log_file="$exp_dir/log/settings.txt"
echo "data fol: $data_fol" >  $log_file
echo "train_file: $train_file" >> $log_file
echo "test_file: $test_file" >> $log_file
echo "ngram: $ngram" >> $log_file
echo "lang: $lang" >> $log_file
echo "dsis: $dsis" >> $log_file
