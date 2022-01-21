#!/usr/bin/env python
# -*- coding: utf8 -*-

'''Fix macro-average line of classification report: if no docs in test set, do not count F-score'''

import argparse
from utils import write_to_file
from config import dsis


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", required=True, type=str,
                        help="Input file with crawled text")
    parser.add_argument("-o", "--output_file", default='', type=str,
                        help="Output file with cleaned data. Default: add .fix to -i")
    args = parser.parse_args()
    return args


def average_by_idx(idx, in_list):
    '''Average a list of lists by by index'''
    nums = [float(sub[idx]) for sub in in_list]
    avg = round(float(sum(nums)) / float(len(nums)), 3)
    return avg


if __name__ == '__main__':
    args = create_arg_parser()
    new_lines, cur_scores = [], []
    skipped = False
    for line in open(args.input_file, 'r'):
        if line.strip() and line.strip().split()[0] in dsis:
            # Ignore lines with 0 supports
            if not line.strip().split()[4] == "0":
                cur_scores.append(line.strip().split()[1:4])
            else:
                skipped = True
        elif line.strip() and line.strip().startswith("macro avg"):
            # Add extra line with actual macro avg
            prec = average_by_idx(0, cur_scores)
            rec = average_by_idx(1, cur_scores)
            f1 = average_by_idx(2, cur_scores)
            cur_scores = []
            new_lines.append(f"\tFair macro avg         {prec}     {rec}     {f1}")
        new_lines.append(line.rstrip())
    # Write final output
    out_file = args.output_file if args.output_file else args.input_file + '.fix'

    # Only do this if we indeed found categories with 0 support
    if skipped:
        write_to_file(new_lines, out_file, do_strip=False)
    else:
        print ("No categories had 0 support, no need for this script, do not print to file")
