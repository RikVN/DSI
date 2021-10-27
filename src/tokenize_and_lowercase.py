#!/usr/bin/env python
# -*- coding: utf8 -*-

'''Tokenize the data using SpaCy and then lowercase it. Write to output file'''

import argparse
from utils import write_to_file, setup_spacy_tokenizer

def create_arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input_file", required=True, type=str, help="Input file with text")
	parser.add_argument("-o", "--output_file", required=True, type=str, help="Output file")
	parser.add_argument("-l", "--lang", default='en', type=str,
						help="Iso code of language we are cleaning (default en)")
	parser.add_argument("-nt", "--no_tokenize", action="store_true",
						help="Do not perform tokenization, only lowercasing")
	parser.add_argument("-nl", "--no_lower", action="store_true",
						help="Do not perform lowercasing, only tokenization")
	args = parser.parse_args()
	if args.no_tokenize and args.no_lower:
		raise ValueError("Skipping both lowercasing and tokenizing does not make sense")
	return args


def tokenize_and_lowercase(in_file, no_tokenize, no_lower, lang):
	'''Do tokenization and lowercasing (or skip one of the two) for lines in an input file'''
	# First set up correct tokenizer
	tokenizer = setup_spacy_tokenizer(lang)
	new_lines = []
	# Loop over lines and perform tokenization/lowercasing
	for line in open(in_file, 'r'):
		line = line.strip()
		if not no_tokenize:
			line = " ".join([tok.text for tok in tokenizer(line)])
		if not no_lower:
			line = line.lower()
		new_lines.append(line)
	return new_lines


if __name__ == '__main__':
	args = create_arg_parser()
	lines = tokenize_and_lowercase(args.input_file, args.no_tokenize, args.no_lower, args.lang)
	write_to_file(lines, args.output_file)
