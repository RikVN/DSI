#!/usr/bin/env python
# -*- coding: utf8 -*-

'''Select all sentences with a certain confidence for (a) certain DSI(s)'''

import argparse
from tabulate import tabulate
from scipy.special import softmax
from config import dsis as long_dsis
from config import small_dsis
from utils import write_to_file


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pred_file", required=True, type=str,
                        help="File with predictions, including header")
    parser.add_argument("-s", "--sent_file", type=str,
                        help="Sents corresponding to predictions. If not added, just print stats")
    parser.add_argument("-o", "--output_folder", type=str,
                        help="Out folder (names we get automatically) to write selected sents to")
    parser.add_argument("-d", "--dsis", nargs="*", type=str,
                        help="DSIs you want to select. Not specified means all, except other")
    parser.add_argument("-min", "--min_confidence", type=float,
                        help="Model confidence should be at least this value. Not needed for stats")
    parser.add_argument("-max", "--max_confidence", type=float, default=1.0,
                        help="Model confidence should be at most this value. \
                              Default is 1, usually not needed to change it")
    parser.add_argument("-pf", "--print_format", default="both", choices=["both", "pct", "value"],
                        help="For statistics: print percentages, values, or both")
    parser.add_argument("-tf", "--table_format", default="plain",
                        help="For statistics: for Latex, use '-tf latex'")
    parser.add_argument("-sm", "--softmax", action="store_true",
                        help="First do softmax over the output, usually already done so not needed")
    args = parser.parse_args()
    # Validate args
    if args.sent_file and not args.min_confidence:
        raise ValueError("A sentence file was specified but not a min confidence")
    return args


def read_preds(pred_file, do_softmax):
    '''Read predictions and save to list with floats'''
    lines = []
    for idx, line in enumerate(open(pred_file, 'r')):
        if idx == 0:
            dsis = [x.strip() for x in line.split(',')]
        else:
            vals = [float(x) for x in line.split()]
            if do_softmax:
                vals = list(softmax(vals))
            lines.append(vals)
    return [dsi.lower() for dsi in dsis], lines


def select_pred_by_conf(lines, dic, dsis, confidences):
    '''Select prediction for a certain confidence, for all DSIs'''
    # Loop over all the values and select predictions based on conf
    for line in lines:
        for dsi_idx, value in enumerate(line):
            for conf_idx, conf in enumerate(confidences):
                if float(value) >= conf:
                    # Found a value with enough confidence, save for this DSI
                    dic[dsis[dsi_idx]][conf_idx] += 1
                else:
                    # Can break already because confidence is listed ascendingly
                    break
    return dic


def print_stats(dsis, lines, table_format, print_format):
    '''Print how many dsis we have left given a certain confidence'''
    # If confidence is set, use that, else table
    confidences = [0.3, 0.5, 0.7, 0.8, 0.9]

    # Calculate statistics
    dic = {}
    for dsi in dsis:
        dic[dsi] = [0 for i in range(len(confidences))]

    # Save the predictions that have higher than certain confidence in a dictionary
    dic = select_pred_by_conf(lines, dic, dsis, confidences)

    # Create format we want to print
    if print_format == "both":
        values = [[dsi] + [f"{num:,}" for num in dic[dsi]] for dsi in dsis]
        pcts = [[dsi] + [round(float(val) / float(len(lines)) * 100, 2) for val in dic[dsi]] for dsi in dsis]
        # One column format for both, with percentages in brackets
        tab = []
        for idx, dsi in enumerate(dsis):
            col = []
            for val_idx, (val, pct) in enumerate(zip(values[idx], pcts[idx])):
                if val_idx == 0:
                    # DSI has no percentage
                    col.append(val)
                else:
                    col.append(f"{val} ({pct}%)")
            tab.append(col)
    elif print_format == "pct":
        tab = [[dsi] + [round(float(val) / float(len(lines)) * 100, 1) for val in dic[dsi]] for dsi in dsis]
    elif print_format == "value":
        tab = [[dsi] + [f"{num:,}" for num in dic[dsi]] for dsi in dsis]
    # Sort table alphabetically
    sort_tab = sorted(tab, key=lambda x: x[0])
    # Nicely print the scores per DSI
    print(tabulate(sort_tab, tablefmt=table_format, headers = [''] + confidences))


def get_preds_for_conf(lines, sents, dsis, use_dsis, min_conf, max_conf):
    '''Get all precictions that match the confidence threshold, save per individual dsi'''
    dic = {dsi:[] for dsi in dsis}
    # Loop over all predictions
    for line_idx, line in enumerate(lines):
        # Loop over DSIs we are interested, the index based on dsis
        for ds in use_dsis:
            dsi_index = dsis.index(ds)
            # Check confidence of this prediction and add if it's OK
            if line[dsi_index] >=  min_conf and line[dsi_index] <= max_conf:
                # Save [sent, line_idx, conf, dsi]
                dic[ds].append([sents[line_idx], str(line_idx), str(line[dsi_index]), ds])
    return dic


def check_and_validate_dsis(dsis, spec_dsis):
    '''Check and validate the specified DSIs: maybe an abbreviation was used, we can fix that'''
    use_dsis = []
    # Use lowercase only to avoid such issues
    sml_dsis = [dsi.lower() for dsi in small_dsis]
    spc_dsis = [dsi.lower() for dsi in spec_dsis]
    # Loop over specified DSI and see if we can match it with the DSIs from the prediction file
    for dsi in spc_dsis:
        if dsi in dsis:
            # Standard case here, specified DSI was indeed used for predicting
            use_dsis.append(dsi)
        elif dsi in sml_dsis:
            # Abbreviation was used, use the small and long dsi names to find the correct dsi to use
            use_dsis.append(dsis[dsis.index(long_dsis[sml_dsis.index(dsi)])])
        elif dsi in long_dsis:
            raise ValueError(f"Something odd is going on: your specified DSI {dsi} is actually a DSI in the config file, but it was not used during prediction. Please make sure you used the correct files/models")
        else:
            raise ValueError(f"Specfied DSI {dsi} was not used during prediction and is also unknown to us")
    return use_dsis


def select_and_write_predictions(lines, dsis, spec_dsis, sent_file, out_folder, min_conf, max_conf):
    '''Select predictions based on the min and max confidence'''
    # Get original sentences
    sents = [x.strip() for x in open(sent_file ,'r')]

    # Check and validate the specified DSIs: maybe an abbreviation was used, we can fix that
    # Only needed if we actually specified them, otherwise use all
    use_dsis = check_and_validate_dsis(dsis, spec_dsis) if spec_dsis else [dsi for dsi in dsis if dsi != "other"]

    # Get predictions per DSI and save in dictionary dic[dsi] = [[sent, line_idx, conf, dsi], ...]
    dic = get_preds_for_conf(lines, sents, dsis, use_dsis, min_conf, max_conf)

    # Now write the output: two files per DSI, one for just the text for convenience, and one
    # that contains the sentence, original index in prediction file, confidence and DSI
    for dsi in dic:
        # Select the sentences only for this dsi
        snts = [x[0] for x in dic[dsi]]
        # Create tab separated string with all info
        info = ["\t".join(x) for x in dic[dsi]]
        # Create files in output folder based on DSI names
        base = out_folder + "/" +  dsi
        if snts:
            write_to_file(snts, base + '.txt')
            write_to_file(info, base + '.info')
            # For logging purposes
            print (f"Wrote {len(snts)} sentences to {base}.txt and {base}.info")


def main():
    '''Main function for selecting predictions based on a certain min confidence'''
    args = create_arg_parser()
    # Read predictions, first one is header (important!)
    dsis, lines = read_preds(args.pred_file, args.softmax)

    # Now if a sentence file is specified, we use it to select the sentences we want to keep
    if args.sent_file:
        select_and_write_predictions(lines, dsis, args.dsis, args.sent_file, args.output_folder,
                                     args.min_confidence, args.max_confidence)
    else:
        # Otherwise we just print the statistics in a nice table
        print_stats(dsis, lines, args.table_format, args.print_format)


if __name__ == '__main__':
    main()
