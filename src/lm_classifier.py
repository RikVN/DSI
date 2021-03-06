#!/usr/bin/env python

'''Fine-tune a language model for DSI classification'''

import sys
import os
import random as python_random
import argparse
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, \
                         Trainer, TrainingArguments, EarlyStoppingCallback
from basic_classifiers import get_data, read_test_data
from utils import load_json_dict


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_file", required=True, type=str,
                        help="Input file to learn from")
    parser.add_argument("-d", "--dev_file", type=str, required=True,
                        help="Separate dev set to evaluate on")
    parser.add_argument("-of", "--output_folder", type=str, required=True,
                        help="Write output(s) to this folder")
    parser.add_argument("-lm", "--lm_ident", type=str, default="bert-base-uncased",
                        help="Language model identifier (default bert-base-uncased)")
    parser.add_argument("-l", "--limit_train", default=0, type=int,
                        help="Limit training set to this amount of instances (default 0 means no limit)")
    parser.add_argument("-ds", "--down_sample", default=0, type=int,
                        help="Downsample categories to this amount of instances (default 0 means no limit)")
    parser.add_argument("-ao", "--also_other", action="store_true",
                        help="Also downsample the 'other' category. The default is that other is \
                              the same size as the rest of the data combined")
    parser.add_argument("-fc", "--filter_categories", nargs="*", default=[],
                        help="Filter the given categories from the data sets, both train and test")
    # Arguments for training a model
    parser.add_argument("-pa", "--padding", default="max_length", type=str,
                        help="How to do the padding: max_length (default) or longest")
    parser.add_argument("-ml", "--max_length", default=None,
                        help="Max length of the inputs, usually not necessary to specify")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Where we save models and output files")
    parser.add_argument("-str", "--strategy", type=str, choices=["no", "steps", "epoch"],
                        default="epoch", help="Strategy for evaluating/saving/logging")
    parser.add_argument("-bs", "--batch_size", default=12, type=int,
                        help="Batch size per device")
    parser.add_argument("-lr", "--learning_rate", default=1e-5, type=float,
                        help="Learning rate")
    parser.add_argument("-wd", "--weight_decay", default=0, type=float,
                        help="Weight decay")
    parser.add_argument("-mgn", "--max_grad_norm", default=1, type=float,
                        help="Max grad norm")
    parser.add_argument("-ne", "--num_train_epochs", default=2, type=int,
                        help="Training epochs")
    parser.add_argument("-wr", "--warmup_ratio", default=0.1, type=float,
                        help="Warmup ratio (0 to 1)")
    parser.add_argument("-af", "--adafactor", action="store_true",
                        help="Use Adafactor instead of AdamW")
    parser.add_argument("-ls", "--label_smoothing", default=0.1, type=float,
                        help="Label smoothing percentage, 0-1")
    parser.add_argument("-eas", "--eval_accumulation_steps", default=500, type=int,
                        help="Number of steps to accumulate during evaluation")
    parser.add_argument("-gc", "--grad_check", action="store_true",
                        help="Use gradient checkpointing (slower, but more memory efficient)")
    parser.add_argument("-dr", "--dropout", default=0.1, type=float,
                        help="Dropout applied to the classifier layer")
    # Random seed is cmd line arg
    parser.add_argument("-s", "--seed", default=1234, type=int,
                        help="Random seed")
    args = parser.parse_args()
    # Make reproducible as much as possible
    np.random.seed(args.seed)
    python_random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    return args


def get_training_arguments(args, model_dir):
    '''Load all training arguments here. There are a lot more not specified, check:
    https://github.com/huggingface/transformers/blob/master/src/transformers/training_args.py#L72'''
    return TrainingArguments(model_dir, evaluation_strategy=args.strategy,
           per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
           learning_rate=args.learning_rate, weight_decay=args.weight_decay,
           max_grad_norm=args.max_grad_norm, num_train_epochs=args.num_train_epochs,
           warmup_ratio=args.warmup_ratio, logging_strategy=args.strategy, save_strategy=args.strategy,
           seed=args.seed, load_best_model_at_end=True, label_smoothing_factor=args.label_smoothing,
           adafactor=args.adafactor, gradient_checkpointing=args.grad_check,
           eval_accumulation_steps=args.eval_accumulation_steps, log_level="debug",
           metric_for_best_model='f1', save_total_limit=1)


class DSIDataset(torch.utils.data.Dataset):
    '''Dataset for using Transformers'''
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    '''Compute the metrics we are interested in'''
    labels = pred.label_ids
    # Sometimes the output is a tuple, take first argument then
    if isinstance(pred.predictions, tuple):
        pred = pred.predictions[0]
    else:
        pred = pred.predictions
    preds = pred.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


def find_model_dir(folder):
    '''In an output dir, automatically find the latest checkpoint dir'''
    subfolders = [f.name for f in os.scandir(folder) if f.is_dir() and f.name.startswith('checkpoint')]
    fol_nums = [[fol, int(fol.split("-")[-1])] for fol in subfolders]
    sort = sorted(fol_nums, key=lambda x: x[1], reverse=True)
    return folder + "/" + sort[0][0] + "/"


def get_uniq_labels(in_list):
    '''Get consistent list of labels from input'''
    keep = []
    for item in in_list:
        if item not in keep:
            keep.append(item)
    return keep


def process_data(in_file, filter_cats, down_sample, also_other, limit_train,
                 lm_ident, padding, max_length, uniq_labels):
    '''Process a data set to get ready for training/testing a model'''
    # Read in data
    if not uniq_labels:
        X_data, Y_data = get_data(in_file, filter_cats, down_sample, also_other, limit_train)
    else:
        X_data, Y_data = read_test_data(in_file, filter_cats)

    # Tokenize the data here
    tokenizer = AutoTokenizer.from_pretrained(lm_ident)
    if max_length is not None:
        max_length = int(max_length)
    data_inputs = tokenizer(X_data, max_length=max_length, padding=padding, truncation=True)

    # Log the maximum input length
    print ("Maximum input length found: {0}\n".format(max([len(val) for val in data_inputs["input_ids"]])))

    # Determine the labels in an order we can get back if needed
    if not uniq_labels:
        uniq_labels = get_uniq_labels(Y_data)
        print ("Labels:\n{0}\n".format(", ".join(uniq_labels)))

    # Convert labels to numbers to avoid errors
    Y_data = [uniq_labels.index(x) for x in Y_data]
    # Make it a dummy data set for the test set to avoid errors
    Y_data_use = [0 for _ in X_data] if not Y_data else Y_data

    # Make it a Dataset (necessary)
    data = DSIDataset(data_inputs, Y_data_use)
    return data, Y_data, uniq_labels


def main():
    '''Main function to train and test neural network given cmd line arguments'''
    args = create_arg_parser()
    # Read in data for training
    train_data, Y_train, uniq_labels = process_data(args.train_file, args.filter_categories,
                                       args.down_sample, args.also_other, args.limit_train,
                                       args.lm_ident, args.padding, args.max_length, [])
    # Read in dev data
    dev_data, _, _ = process_data(args.dev_file, args.filter_categories, args.down_sample,
                                      args.also_other, args.limit_train, args.lm_ident,
                                      args.padding, args.max_length, uniq_labels)

    # Setup variables
    num_labels = len(set(Y_train))
    model_dir = args.output_dir

    # Select model we will use
    config = AutoConfig.from_pretrained(args.lm_ident, classifier_dropout=args.dropout, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(args.lm_ident, config=config)

    # Setup arguments training the model
    training_args = get_training_arguments(args, model_dir)

    # For logging purposes
    print("Generated by command:\npython", " ".join(sys.argv))
    print("Logging training settings\n", training_args)

    # Set EarlyStopping with patience of 1, since we evaluate on F1 each epoch and have lots of data
    callbacks = [EarlyStoppingCallback(early_stopping_patience=1)]
    trainer = Trainer(model=model, args=training_args, train_dataset=train_data,
                      eval_dataset=dev_data, compute_metrics=compute_metrics, callbacks=callbacks)

    # Actually train here, best model gets saved automatically
    mets = trainer.train()
    
    # Log the training info and we're done
    print("\nTraining info:\n", mets,"\n")
	

if __name__ == '__main__':
    main()
