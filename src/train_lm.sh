#!/bin/bash
# Job scheduling info, only for us specifically
#SBATCH --time=1-23:59:59
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=50G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rikvannoord@gmail.com
set -eu -o pipefail

# We read the settings from two files: a file with default settings
# and a file with our specific experimental settings (override) that is the argument in $1
source configs/default.sh
source $1

# Set main directory: in our experiments this always works,
# configs are in configs/, experiments in exps/
base_dir=$(dirname "$1")
main_dir=$(echo ${base_dir}/ | sed 's/configs/exps/')
filename=$(basename -- "$1")
filename="${filename%.*}"
out_fol="$main_dir${filename}/"

# Setup folder structure
for nme in log models output eval bkp; do
	mkdir -p ${out_fol}/${nme}
done

# Training call
python src/lm_classifier.py --train_file $train_file --dev_file $dev_file -lm $lm_ident -l $limit_train -ds $down_sample $also_other $filter_categories -o ${out_fol}/models/ -str $strategy -bs $batch_size -lr $learning_rate -wd $weight_decay -mgn $max_grad_norm -ne $num_train_epochs -wr $warmup_ratio $adafactor -ls $label_smoothing -of ${out_fol}/output/ -pa $padding $grad_check -eas $eval_accumulation_steps $max_length --dropout $dropout --seed $seed > ${out_fol}/log/train.log 2> ${out_fol}/log/stderr.log

# Do parsing for specified test files, save eval files
count=0
for test_file in $files_to_parse; do
	echo "Producing output for $test_file"
	echo "Writing to ${out_fol}/output/${out_prefixes[$count]}"
	# We save only 1 checkpoint so we can select it like this
	python src/lm_parse.py -m ${out_fol}/models/ --train_log ${out_fol}/log/train.log --lm_ident $lm_ident --sent_file $test_file -o ${out_fol}/output/${out_prefixes[$count]} -pa $padding > ${out_fol}/eval/${out_prefixes[$count]}.eval 2> ${out_fol}/log/${out_prefixes[$count]}.log
	# For Dutch/Spanish we fix the evaluation file macro average (don't count categories with 0 support)
	python src/fix_clf_report.py -i ${out_fol}/eval/${out_prefixes[$count]}.eval
	(( count++ )) || true
done

# Backup for reproducibility: the config and default files, just to be sure
cp src/default.sh ${out_fol}/bkp/
cp $1 ${out_fol}/bkp/


