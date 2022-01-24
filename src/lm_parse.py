#!/usr/bin/env python

'''Make predictions using a trained finetuned LM on a file of sentences

   Example usage:
   python lm_parse.py --model model/ --sent_file sentences.txt --lang en'''

import sys
import os
import random as python_random
import argparse
import ast
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, Trainer
from scipy.special import softmax
from sklearn.metrics import classification_report
from lm_classifier import process_data
from utils import write_to_file


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="Location of the trained model (folder)")
    parser.add_argument("-s", "--sent_file", required=True, type=str,
                        help="Predict on these sentences, not tokenized/processed yet")
    parser.add_argument("-l", "--lang", type=str,
                        help="Iso code of lang we are parsing, we use this to select lm_ident if not specified")
    parser.add_argument("-o", "--output_file", type=str,
                        help="Output file, if not specified add .pred and .pred.prob to sent file")
    parser.add_argument("-lm", "--lm_ident", type=str,
                        help="Language model identifier. No need to specify it, unless you trained \
                              your own model that is different from ours")
    parser.add_argument("-pa", "--padding", default="max_length", type=str,
                        help="How to do the padding: max_length (default) or longest")
    parser.add_argument("-se", "--seed", default=2345, type=int,
                        help="Should not matter for prediction, but you can specify it either way")
    # HuggingFace does not allow to only keep a single model throughout training, so you can
    # always resume training. But this is annoying if you really never plan to do that anyway
    # We want to automatically determine the model we use for parsing (if there are two)
    # For that to work, we read the log file and the corresponding metrics. If the best metric is
    # not the last model, we use the first one. We have to read this from a log file, as
    # HuggingFace apparantely does not easily return metrics per epoch in an object
    parser.add_argument("-tl", "--train_log", type=str,
                        help="Location of train log file if we have to find the model")
    args = parser.parse_args()

    # Validate arguments
    if not args.lang and not args.lm_ident:
        raise ValueError("Specify at least one of --lang or --lm_ident")

    # The seed shouldn't matter for only parsing (and doesn't in our experiments)
    # But we set it anyway here so you can experiment with it if you want
    np.random.seed(args.seed)
    python_random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    return args


def get_best_epoch_idx(train_log):
    '''Return 1 if highest epoch was the best (F1 based), else 0'''
    best_f, best_idx, total = 0, 0, 0
    # Loop over the log file and read all the eval_loss lines as a dict
    for line in open(train_log, 'r'):
        if line.strip()[1:].startswith("'eval_loss':"):
            dic = ast.literal_eval(line.strip())
            if dic["eval_f1"] > best_f:
                best_f = dic["eval_f1"]
                best_idx = total
            total += 1
    # Best idx was the last one
    if best_idx == total -1:
        return 1
    # Else it was the previous one, so the only one we have left
    else:
        return 0


def select_model(mod, train_log):
    '''Select highest or lowest checkpoint based on the logs'''
    subfolders = [f.name for f in os.scandir(mod) if f.is_dir() and f.name.startswith('checkpoint')]
    # No checkpoints, just work with this model
    if not subfolders:
        return mod
    elif len(subfolders) == 1:
        # Just return the one model we found
        return mod + "/" + subfolders[0] + "/"
    elif len(subfolders) > 2:
        raise ValueError("If you do not specify an actual single model, we can only work with a folder of two checkpoints")
    # Sort checkpoints from low to high
    fol_nums = [[fol, int(fol.split("-")[-1])] for fol in subfolders]
    sort = sorted(fol_nums, key=lambda x: x[1], reverse=False)
    # Read the log file and decide if highest epoch was best
    if not train_log:
        raise ValueError("You need to specify --train_log if you want to automatically find the best model in args.model")
    idx = get_best_epoch_idx(train_log)
    return mod + "/" + sort[idx][0] + "/"


def evaluate(trainer, in_data, output_file, Y_data, uniq_labels, do_softmax):
    '''Evaluate a trained model on a dev/test set, print predictions to file possibly'''
    # Actually get the output
    output = trainer.predict(in_data)
    # ometimes the output is a tuple, take first argument then
    if isinstance(output.predictions, tuple):
        out = output.predictions[0]
    else:
        out = output.predictions

    preds = np.argmax(out, axis=1)
    header = [", ".join(uniq_labels)]
    # If a file was specified, print predictions to this file
    # First convert numbers back to labels
    if output_file:
        out_labels = [uniq_labels[pred] for pred in preds]
        write_to_file(out_labels, output_file)
        # Write probabilities, maybe do softmax first
        if do_softmax:
            out_lines = header + [" ".join([str(x) for x in softmax(row)]) for row in out]
        else:
            out_lines = header + [" ".join([str(x) for x in row]) for row in out]
        write_to_file(out_lines, output_file + '.prob')

    # Print classification report if we have labels
    if Y_data:
        # For a nicer report, convert labels back to strings first
        Y_lab = [uniq_labels[idx] for idx in Y_data]
        pred_lab = [uniq_labels[idx] for idx in preds]
        print ("Classification report:\n")
        print (classification_report(Y_lab, pred_lab, digits=3))


def main():
    '''Main function to parse a new file with a finetuned LM given cmd line arguments'''
    args = create_arg_parser()
    # LMs we trained models for
    lm_en = "microsoft/deberta-v3-large"
    lm_ml = "xlm-roberta-large"

    # If identifier is not specified we select it based on the language
    if not args.lm_ident:
        lm_ident = lm_en if args.lang.lower() in ['en', 'english', 'eng'] else lm_ml
    else:
        lm_ident = args.lm_ident

    # Set order of labels (important!), as this was automatically determined during training,
    # so we have to use the same order for the predictions to make sense
    labels = ["open-data-portal", "other", "europeana", "e-justice", "online-dispute-resolution",
              "cybersecurity", "safer-internet", "eessi", "e-health"]

    # Max length is important to set, as we find it during training the models, but is not
    # saved as an intrinsic part of the model
    # It looks like we get similar results by using 512 anyway, but to be sure we set it here
    if lm_ident in [lm_en, lm_ml]:
        max_length = 177
    else:
        print("WARNING: model must is not deberta or xlm-roberta, set max_length to default 512")
        max_length = 512

    # Read in data for dev and test. Lots of arguments can be default/empty/false, they only matter
    # for training a model and since we use the same function we have to specify them here
    test_data, Y_test, _ = process_data(args.sent_file, [], 0, False, 0, lm_ident,
                                        args.padding, max_length, labels)

    # Select model, see explanation in argparser as to why this complicated procedure is necessary
    mod = select_model(args.model, args.train_log)
    print (f"Do prediction with {mod}")
    # Set up the model and the trainer
    model = AutoModelForSequenceClassification.from_pretrained(mod)
    trainer = Trainer(model=model)

    # If we didn't specify the output file, add .pred and .pred.prob to the sentence file
    if not args.output_file:
        out_file = args.sent_file + '.pred'
    else:
        out_file = args.output_file

    # Run evaluation, this write the predictions to an output file (Y_test is empty)
    # Note that in outfile.prob you get the softmax predictions!
    # We do softmax here as a default, but you could also get the logits (by True -> False)
    # If your sent file contains labels (tab separated), we automatically evaluate the predictions
    evaluate(trainer, test_data, out_file, Y_test, labels, True)


if __name__ == '__main__':
    # For logging purposes
    print("Generated by command:\npython", " ".join(sys.argv))
    main()
