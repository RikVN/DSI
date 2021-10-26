#!/usr/bin/env python
# -*- coding: utf8 -*-

'''Print a table with all perplexities per DSI'''

import argparse
import os
from tabulate import tabulate
from config import dsis, small_dsis

def create_arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input_folder", required=True, type=str,
						help="Main input folder")				
	parser.add_argument("-l", "--lang", default='en', type=str,
						help="Iso code of language we are looking at")	
	args = parser.parse_args()
	return args


def get_ppl_from_file(in_file):
	'''Read file and return perplexity (excluding OOV)'''
	lines = [x.strip() for x in open(in_file, 'r')]
	for line in lines:
		if line.startswith("Perplexity excluding OOVs:"):
			val = str(round(float(line.strip().split()[-1]), 1))
			return val
	return ValueError("Perplexity not found in {0}".format(in_file))

if __name__ == '__main__':
	args = create_arg_parser()
	# We assume run_kenlm.sh has been run, so we can loop over the files it created
	full_list = []
	for dsi in dsis:
		cur_fol = args.input_folder + dsi + '/' + args.lang + '/'
		cur_list = [dsi]
		for eval_dsi in dsis:
			eval_file = cur_fol + eval_dsi + '.eval'
			ppl = get_ppl_from_file(eval_file)
			cur_list.append(ppl)
		full_list.append(cur_list)
	# Print list of perplexities
	print(tabulate(full_list, headers=[''] + small_dsis))
