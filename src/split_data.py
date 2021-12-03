#!/usr/bin/env python
# -*- coding: utf8 -*-

'''Split data in train/test for language modelling'''

import argparse
import os
from random import shuffle
from config import dsis
from utils import write_to_file


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", required=True, type=str,
                        help="Main input folder")
    parser.add_argument("-l", "--lang", default='en', type=str,
                        help="Iso code of language we are dealing with (default en)")
    parser.add_argument("-e", "--extension", default='.clean.tok.lower.dedup', type=str,
                        help="Extension of the files we are splitting")
    parser.add_argument("-fa", "--file_add", default='', type=str,
                        help="What we add to the file names of train/dev/test, e.g. _clf for train_clf.ext")
    parser.add_argument("-s", "--shuffle", action="store_true",
                        help="Whether we shuffle the data before splitting")
    parser.add_argument("-d", "--dev_size", default=1000, type=int,
                        help="Size of the dev set")
    parser.add_argument("-t", "--test_size", default=1000, type=int,
                        help="Size of the test set")
    parser.add_argument("-al", "--add_label", action="store_true",
                        help="Whether we add the label as second item after the sentence (with tab).")
    parser.add_argument("-mo", "--more_other", action="store_true",
                        help="Add more 'other' instances, by doing: dev/testsize * num_dsis")
    args = parser.parse_args()
    return args


def split_data(input_folder, d_size, t_size, lang, file_ext, do_shuffle, add_label, file_add, more_other):
    '''Given an input folder, find the training files and split it'''
    train_name = "train" + file_add
    dev_name = "dev" + file_add
    test_name = "test" + file_add
    file_name = 'sentences'
    for dsi in dsis:
        # Read in original file
        cur_fol = input_folder + dsi + '/' + lang + '/'
        orig = cur_fol + file_name + file_ext
        if os.path.isfile(orig):
            lines = [x.strip() for x in open(orig, 'r')]
            # Check if we found enough line so that training set is at least 3x dev + test
            if len(lines) > 3 * (d_size + t_size):
                # Shuffle original file if we want
                if do_shuffle:
                    shuffle(lines)
                # Check if we want to add the label to the sentence
                # This is easier for the classification task, where we need to keep this info
                if add_label:
                    lines = [x + '\t' + dsi for x in lines]

                # Dev/test sizes are different for category "other" (maybe)
                dev_size = (len(dsis) - 1) * d_size if more_other and dsi == "other" else d_size
                test_size = (len(dsis) - 1) * t_size if more_other and dsi == "other" else t_size

                # Create splits
                train = lines[:len(lines) - dev_size - test_size]
                dev = lines[len(train): len(train) + dev_size]
                test = lines[len(train) + dev_size:]
                # Write to file
                write_to_file(train, cur_fol + train_name + file_ext)
                write_to_file(dev, cur_fol + dev_name + file_ext)
                write_to_file(test, cur_fol + test_name + file_ext)
            else:
                print ("Skipping {0}, length of {1} too small".format(orig, len(lines)))
        else:
            print ("Skipping {0}, file does not exist".format(orig))


if __name__ == '__main__':
    args = create_arg_parser()
    # Split data sets here
    split_data(args.input_folder, args.dev_size, args.test_size, args.lang, args.extension, args.shuffle, args.add_label, args.file_add, args.more_other)
