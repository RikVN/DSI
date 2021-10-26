#!/bin/bash
# Train a kenLM language model and evaluate on all other files
# Files we train on are in sentences.clean
cur_dir=$(pwd)
kenlm_home="tools/kenlm/build/"
ngram="3"

# Set dsis we are working with
dsis="cybersecurity e-health e-procurement online-dispute-resolution safer-internet ec-europa e-justice europeana open-data-portal"

for dsi in $dsis ; do
	# Select file
	cur_fol="${cur_dir}/data/crawl_19_10_2021/${dsi}/en/"
	cur_file="${cur_fol}/sentences.clean"
	# Train KenLM
	cd $kenlm_home
	./bin/lmplz -o 3 -S 100G -T /tmp < $cur_file > ${cur_fol}sentences.arpa
	# Make model a binary
	./bin/build_binary ${cur_fol}sentences.arpa ${cur_fol}sentences.binary
	# Evaluate on models
	for eval in $dsis ; do
		eval_file="${cur_dir}/data/crawl_19_10_2021/${eval}/en/sentences.clean"
		./bin/query ${cur_fol}sentences.binary < $eval_file > ${cur_fol}/${eval}.eval
	done
	# Back to original dir
	cd $cur_dir
done
