#!/bin/bash
lm_ident="bert-base-multilingual-cased"
dev_file="v1/es-nl/dev"
# We have to re-specify that we want do test on the dev set, since we do not always
# want to test on the dev set, e.g. for Spanish/Dutch
files_to_parse="v1/es/dev v1/nl/dev"
# Important that this is an array, we loop over $test_files and use idx
out_prefixes=( dev_es dev_nl )
# Since this is a test config, downsample quite a bit to speed it up
down_sample="300"
num_train_epochs="1"
