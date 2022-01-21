#!/usr/bin/env python

'''Do evaluation on predicted and gold standard file'''

import argparse
from sklearn.metrics import classification_report, confusion_matrix
from utils import plot_confusion_matrix
from config import to_small
from lm_classifier import get_uniq_labels


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gold_file", type=str, required=True,
                        help="File with gold standard")
    parser.add_argument("-p", "--pred_file", type=str, required=True,
                        help="File with predictions")
    parser.add_argument("-c", "--confusion", default='', type=str,
                        help="Save plot of confusion matrix here, if not added do not plot")
    args = parser.parse_args()
    return args


def main():
    '''Main function for quick classification report evaluation'''
    args = create_arg_parser()

    # Read in data
    pred = [x.strip() for x in open(args.pred_file, 'r')]
    gold = [x.strip() for x in open(args.gold_file, 'r')]
    labels = sorted(get_uniq_labels(pred))
    # Print the report
    print ("Classification report:\n")
    print (classification_report(gold, pred, digits=3))

    # Plot confusion matrix if wanted
    if args.confusion:
        names = [to_small[lab] for lab in labels]
        use_names = []
        # Using ODP instead of Open looks better in the confusion matrix
        for name in names:
            if name == "Open":
                use_names.append("ODP")
            else:
                use_names.append(name)
        plot_confusion_matrix(confusion_matrix(gold, pred), use_names, args.confusion, normalize=False)


if __name__ == '__main__':
    main()
