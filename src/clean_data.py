#!/usr/bin/env python
# -*- coding: utf8 -*-

'''Clean the DSI data to only contain sentences (and not headers, links, etc) of the correct language'''

import argparse
from spacy.lang.en import English


def create_arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input_file", required=True, type=str,
						help="Input file with crawled text")
	parser.add_argument("-ml", "--min_length", default=5, type=int,
						help="Minimum sentence length to be included")
	parser.add_argument("-o", "--output_file", default='', type=str,
						help="Output file with cleaned data. Default: add .clean to -i")
	args = parser.parse_args()
	return args


def length_check(line, tokenizer, min_length):
	'''Check if the sentence passed the length check'''
	return len(tokenizer(line)) >= min_length


def ends_with_punctuation(line):
	return line[-1] in ['.', '?', '!', ':', ';']


def filter_doubles_from_list(items):
	'''Filter double items in a list but keep order'''
	return list(dict.fromkeys(items))


def clean_data(input_file, min_length):
	'''Clean the DSI data to only contain sentences/texts we want to keep'''
	# Setup spacy tokenization for later checks
	nlp = English()
	tokenizer = nlp.tokenizer
	
	# Now loop over the input sentences/texts and check if we keep them
	keep_texts = []
	for idx, line in enumerate(open(input_file, 'r')):
		line = line.strip()
		# First do a simple length check, which requires tokenization
		if not length_check(line, tokenizer, min_length):
			continue
		# Sentences should end with punctuation of some sort
		if not ends_with_punctuation(line):
			continue	
		keep_texts.append(line)
	# We also want to normalize: quotes, dashes, etc
	# To be added here
	
	# Filter double lines from the ones we keep
	keep_texts = filter_doubles_from_list(keep_texts)
	print ("{0} of {1} lines remain".format(len(keep_texts), idx+1))	
		

if __name__ == '__main__':
	args = create_arg_parser()
	clean_lines = clean_data(args.input_file, args.min_length)
