#!/usr/bin/env python
# -*- coding: utf8 -*-

'''Clean the DSI data to only contain sentences (and not headers, links, etc) of the correct language'''

import argparse
import cld3
from config import punctuation, replacements
from utils import write_to_file, setup_spacy_tokenizer


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", required=True, type=str,
                        help="Input file with crawled text")
    parser.add_argument("-minl", "--min_length", default=7, type=int,
                        help="Minimum sentence length to be included")
    parser.add_argument("-maxl", "--max_length", default=50, type=int,
                        help="Minimum sentence length to be included")
    parser.add_argument("-o", "--output_file", default='', type=str,
                        help="Output file with cleaned data. Default: add .clean to -i")
    parser.add_argument("-l", "--lang", default='en', type=str,
                        help="Iso code of language we are cleaning")
    parser.add_argument("-on", "--only_normalize", action="store_true",
                        help="Only normalize the input data, no other filtering steps.")
    args = parser.parse_args()
    return args


def length_check(line, tokenizer, min_length, max_length):
    '''Check if the sentence passed the length check'''
    tok_len = len(tokenizer(line))
    return min_length <= tok_len <= max_length


def ends_with_punctuation(line):
    '''Check string ends with something we consider punctuation, see config.py'''
    return line[-1] in punctuation


def filter_doubles_from_list(items):
    '''Filter double items in a list but keep order'''
    return list(dict.fromkeys(items))


def normalize(line):
    '''Normalize a string'''
    # Resolve multiple spaces
    line = " ".join(line.split())
    # Pairs of replacements here, for quotes/dashes
    # See config.py for pairs of replacements
    for char, replace_with in replacements:
        line = line.replace(char, replace_with)
    return line


def correct_language(line, lang):
    '''Check if a certain string belongs to the language we specified
       Use pycld3: https://github.com/bsolomon1124/pycld3'''
    res = cld3.get_language(line)
    return res.language == lang


def specific_filter(line):
    '''Data-specific things we found that we want to filter as well, not based on general principles'''
    # The data of cybersecurity contains hundreds of these:
    # No matching events listed under Workshops scheduled for DATE
    # Just filter them
    if line.startswith("No matching events listed under"):
        return True
    return False


def clean_data(input_file, min_length, max_length, lang, only_normalize):
    '''Clean the DSI data to only contain sentences/texts we want to keep'''
    # Setup spacy tokenization for later checks
    tokenizer = setup_spacy_tokenizer(lang)

    # Now loop over the input sentences/texts and check if we keep them
    keep_texts = []
    for idx, line in enumerate(open(input_file, 'r')):
        #if idx % 100000 == 0:
        #    print (idx)
        line = line.strip()
        # First normalize the input: quotes, dashes, etc
        line = normalize(line)
        # If we only do normalization we are already done
        if not only_normalize:
            # Do a simple length check, which requires tokenization
            if not length_check(line, tokenizer, min_length, max_length):
                continue
            # Sentences should end with punctuation of some sort
            if not ends_with_punctuation(line):
                continue
            # The sentence should also be in the correct language (e.g. noticed some Russian for English)
            if not correct_language(line, lang):
                continue
            # Data-specific things we found that we want to filter as well
            if specific_filter(line):
                continue
        # Keep the final line
        keep_texts.append(line)

    # Filter double lines as well
    keep_texts = filter_doubles_from_list(keep_texts)
    print ("{0} of {1} lines remain for {2}".format(len(keep_texts), idx+1, input_file))
    return keep_texts


if __name__ == '__main__':
    args = create_arg_parser()
    # Clean the data here
    clean_lines = clean_data(args.input_file, args.min_length, args.max_length, args.lang, args.only_normalize)
    # Write them to output file, add .clean to input if not specified
    out_file = args.output_file if args.output_file else args.input_file + '.clean'
    write_to_file(clean_lines, out_file)
