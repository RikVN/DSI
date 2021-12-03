#!/usr/bin/env python

'''Fine-tune a language model for DSI classification'''

import random as python_random
import argparse
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from basic_classifiers import get_data, read_test_data

# Make reproducible as much as possible
seed = 1234
np.random.seed(seed)
python_random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_file", required=True, type=str,
                        help="Input file to learn from")
    parser.add_argument("-d", "--dev_file", type=str, required=True,
                        help="Separate dev set to evaluate on")
    parser.add_argument("-t", "--test_file", type=str,
                        help="If added, use trained model to predict on this test set")
    parser.add_argument("-lm", "--lm_ident", type=str, default="bert-base-uncased",
                        help="Language model identifier (default bert-base-uncased)")
    parser.add_argument("-l", "--limit_train", default=0, type=int,
                        help="Limit training set to this amount of instances (default 0 means no limit)")
    parser.add_argument("-ds", "--down_sample", default=0, type=int,
                        help="Downsample categories to this amount of instances (default 0 means no limit)")
    parser.add_argument("-ao", "--also_other", action="store_true",
                        help="Also downsample the 'other' category. The default is that other is the same size as the rest of the data combined")
    parser.add_argument("-fc", "--filter_categories", nargs="*", default=[],
                        help="Filter the given categories from the data sets, both train and test")
    parser.add_argument("-tnl", "--test_no_labels", default="",
                        help="The test set has no labels: print predictions to this file")
    parser.add_argument("-pr", "--probabilities", action="store_true",
                        help="Print the probabilities to a file instead of the labels for -tnl")
    args = parser.parse_args()
    return args


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
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


def main():
    '''Main function to train and test neural network given cmd line arguments'''
    args = create_arg_parser()

    # Read in training data
    X_train, Y_train = get_data(args.train_file, args.filter_categories, args.down_sample, args.also_other, args.limit_train)
    X_dev, Y_dev = read_test_data(args.dev_file, False, args.filter_categories)

    # Convert labels to numbers to avoid errors
    uniq_labels = list(set(Y_train))
    num_labels = len(set(Y_train))
    Y_train = [uniq_labels.index(x) for x in Y_train]
    Y_dev = [uniq_labels.index(x) for x in Y_dev]

    # Setup tokenizer and tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.lm_ident)
    train_inputs = tokenizer(X_train, padding="max_length", truncation=True)
    dev_inputs = tokenizer(X_dev, padding="max_length", truncation=True)

    # Transformer to Dataset object
    train_data = DSIDataset(train_inputs, Y_train)
    dev_data = DSIDataset(dev_inputs, Y_dev)

    # Setup model
    model = AutoModelForSequenceClassification.from_pretrained(args.lm_ident, num_labels=num_labels)

    # Training the model
    training_args = TrainingArguments("test_trainer", num_train_epochs=3)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_data, eval_dataset=dev_data, compute_metrics=compute_metrics)
    trainer.train()

    # Now evaluate on the dev set and print classification report
    output = trainer.predict(dev_data)
    preds = np.argmax(output.predictions, axis=1)
    print (classification_report(Y_dev, preds, digits=3))


if __name__ == '__main__':
    main()
