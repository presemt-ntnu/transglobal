#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
print experimental results as text table
"""

import argparse

import numpy

from tg.utils import text_table

def print_results(results_fname, out_fname=None):
    table = numpy.load(results_fname)
    text_table(table, out_fname)


parser = argparse.ArgumentParser(description=__doc__)     
    
parser.add_argument(
    "results_fname",
    metavar="RESULTS_FILE",
    help="results file in .npy format")
    
parser.add_argument(
    "out_fname",
    metavar="OUTPUT_FILE",
    nargs="?",
    help="file for writing output")

args = parser.parse_args()

print_results(args.results_fname, args.out_fname)