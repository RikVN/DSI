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
	parser.add_argument("-t", "--test_size", default=5000, type=int,
						help="Size of the test set, if total data > 2.5 * test size, else half")
	args = parser.parse_args()
	return args


def split_data(input_folder, test_size, lang):
	'''Given an input folder, find the training files and split it'''
	train_name = "train"
	test_name = "test"
	file_name = 'sentences'
	file_ext = '.clean.tok.lower'
	for idx, dsi in enumerate(dsis):
		# Read in original file
		cur_fol = input_folder + dsi + '/' + lang + '/'
		orig = cur_fol + file_name + file_ext
		lines = [x.strip() for x in open(orig, 'r')]
		# Shuffle original file
		shuffle(lines)
		# Check if we change the test size
		cur_test = test_size if not (len(lines) < (float(test_size) * 2)) else int(test_size / 2)
		# Create splits
		train = lines[cur_test:]
		test = lines[:cur_test]
		# Write to file
		write_to_file(train, cur_fol + train_name + file_ext)
		write_to_file(test, cur_fol + test_name + file_ext)


if __name__ == '__main__':
	args = create_arg_parser()
	# Split data sets here
	split_data(args.input_folder, args.test_size, args.lang)

