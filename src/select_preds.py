#!/usr/bin/env python
# -*- coding: utf8 -*-

'''Select all sentences with a certain confidence for a certain DSI'''

import argparse
from tabulate import tabulate


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sent_file", required=True, type=str,
                        help="Main input folder")
    parser.add_argument("-p", "--pred_file", required=True, type=str,
                        help="File with predictions, including header")
    parser.add_argument("-c", "--confidence", default=0.5, type=float,
                        help="Model confidence should be at least this value")
    args = parser.parse_args()
    return args


def read_preds(pred_file):
    '''Read predictions and save to list with floats'''
    lines = []
    for idx, line in enumerate(open(pred_file, 'r')):
        if idx == 0:
            dsis = line.strip().split()
        else:
            lines.append([float(x) for x in line.split()])
    return dsis, lines


def print_stats(dsis, lines, confidence):
    '''Print how many dsis we have left given a certain confidence'''
    dic = {dsi: 0 for dsi in dsis}
    # Loop over all the values
    for line in lines:
        for idx, value in enumerate(line):
            if value >= confidence:
                # Found a value with enough confidence, save for this DSI
                dic[dsis[idx]] += 1
    # Nicely print the scores per DSI
    print ("\nNumber of insts with confidence >= {0} per DSI:\n".format(confidence))
    print(tabulate([[dsi, dic[dsi], round((float(dic[dsi]) / len(lines)) * 100, 1)] for dsi in dsis]))


if __name__ == '__main__':
    args = create_arg_parser()
    # Get original sentences
    sents = [x.strip() for x in open(args.sent_file ,'r')]
    # Now loop over DSIs, first one is header
    dsis, lines = read_preds(args.pred_file)
    # Calculate statistics maybe
    print_stats(dsis, lines, args.confidence)
