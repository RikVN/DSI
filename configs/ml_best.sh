#!/bin/bash
lm_ident="xlm-roberta-large"
dev_file="v1/es-nl/dev"
files_to_parse="v1/es/dev v1/nl/dev"
out_prefixes=( dev_es dev_nl )
down_sample="50000"
num_train_epochs="5"
