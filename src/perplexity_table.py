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
	parser.add_argument("-o", "--oov", choices=["including", "excluding"], type=str,
						default="excluding", help="Get perplexity for including or excluding oov tokens")
	args = parser.parse_args()
	return args


def get_ppl_from_file(in_file, oov_type):
	'''Read file and return perplexity'''
	for line in open(in_file, 'r'):
		if line.strip().startswith("Perplexity {0} OOVs:".format(oov_type)):
			val = str(round(float(line.strip().split()[-1]), 1))
			return val
	return ValueError("Perplexity not found in {0}".format(in_file))


def get_closest_dsis(ppl_list, exs_dsis):
	'''For a given list of perplexities per DSI, order them by how close they are to the current DSI'''
	out_list = []
	sml_dsis = [small_dsis[idx] for idx in exs_dsis]
	for idx, values in enumerate(ppl_list):
		# Select pairs of value-dsi and sort based on first item (value)
		pairs = sorted([[value, dsi] for value, dsi in zip([float(x) for x in values[1:]], sml_dsis)], key=lambda x: x[0])
		# Convert to nice format for printing
		out_list.append([dsis[exs_dsis[idx]]] + ["{0} ({1})".format(dsi, value) for value, dsi in pairs])
	return out_list


def get_perplexity_list(input_folder, oov_type):
	'''Given an input folder with eval files for a DSI, extract the perplexities'''
	# We assume run_kenlm.sh has been run, so we can loop over the files it created
	full_list = []
	existing_dsis = []
	for idx, dsi in enumerate(dsis):
		cur_fol = input_folder + dsi + '/'
		if os.path.isdir(cur_fol):
			existing_dsis.append(idx)
			# Sometimes we run experiments for a subset of dsis
			# Then ignore this dsi, but do keep track of existing dsis
			cur_list = [dsi]
			for eval_dsi in dsis:
				eval_file = cur_fol + eval_dsi + '.eval'
				ppl = get_ppl_from_file(eval_file, oov_type)
				cur_list.append(ppl)
			full_list.append(cur_list)
	return full_list, existing_dsis


if __name__ == '__main__':
	args = create_arg_parser()
	# Get the list of lists with all perplexities per DSI over other DSIs
	ppl_list, exs_dsis = get_perplexity_list(args.input_folder, args.oov)
	# Print the list as a nicely formatted table
	print(tabulate(ppl_list, headers=[''] + small_dsis))
	# For each DSI, print the closest DSIs in terms of perplexity
	close_list = get_closest_dsis(ppl_list, exs_dsis)
	# Print nice table with those values as well
	print ()
	print(tabulate(close_list))
