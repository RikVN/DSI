#!/bin/bash

# Set data variables
train_file="v1/en/train"
dev_file="v1/en/dev"
files_to_parse="v1/en/dev" # add as "file1 file2 file3"
out_prefixes=( dev )

# Process variables
limit_train="0" # 0 means no limit
down_sample="0" # 0 means no down sampling
also_other="" # add as --also_other
filter_categories="" # add as: "-fc other cybersecurity"

# Model and training variables
lm_ident="microsoft/deberta-v3-large" # Specify in your config file a different one for non-English
strategy="epoch"
batch_size="12" # 12 best
learning_rate="1e-5"
weight_decay="0"
max_grad_norm="1"
num_train_epochs="6" # but we use earlystopping with patience 1, so we usually do not get here
warmup_ratio="0.1" # 0.1 best
adafactor="" # add as: --adafactor (did not help in our experiments)
label_smoothing="0.1" # 0.1 best
padding="longest"
grad_check="" # add as --grad_check
eval_accumulation_steps="250"
max_length="" # add as --max_length 512
dropout="0.1"
seed="1234"
