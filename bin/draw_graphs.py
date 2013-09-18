#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Draw translation graphs
"""

import argparse
import cPickle
import logging
import os
import pprint

from tg.draw import Draw
from tg.utils import set_default_log


log = logging.getLogger(os.path.basename(__file__))


def draw_graphs(graphs_fname, out_format, out_dir, i, j, best_score_attr,
                base_score_attrs):
    log.info("Settings:\n" + pprint.pformat(locals()))
    log.info("loading graphs from " + graphs_fname) 
    graphs = cPickle.load(open(graphs_fname))  
    
    draw = Draw()
    draw(graphs[i:j], out_format=out_format, out_dir=out_dir,
         best_score_attr=best_score_attr, base_score_attrs=base_score_attrs)


parser = argparse.ArgumentParser(description=__doc__)     
    
parser.add_argument(
    "graphs_fname",
    metavar="GRAPHS_FILE",
    help="pickle file containing translation graphs")

parser.add_argument(
    "-a", "--base-score-attrs",
    metavar="ATTR",
    nargs="+",
    default=[],
    help="base score attributes")

parser.add_argument(
    "-b", "--best-score-attr",
    metavar="ATTR",
    default="freq_score",
    help="best score attribute (defaults to 'best_score')")

parser.add_argument(
    "-f", "--format",
    metavar="FORMAT",
    default="pdf",
    help="any drawing format supported by Graphviz including "
    "pdf, ps, gif, png and jpeg (defaults to pdf)")

parser.add_argument(
    "-i", "--from",
    dest="i",
    metavar="N",
    type=int,
    default=0,
    help="first graph in range (counting from zero)")

parser.add_argument(
    "-j", "--to",
    dest="j",
    metavar="N",
    type=int,
    default=None,
    help="last graph in range (counting from zero)")

parser.add_argument(
    "-o", "--out-dir",
    metavar="DIR",
    help="output directory (will be created when required)")

parser.add_argument(
    "-v", "--verbose",
    action="store_true")

args = parser.parse_args()

if args.verbose:
    set_default_log()

draw_graphs(args.graphs_fname,
            out_format=args.format,
            out_dir=args.out_dir,
            best_score_attr=args.best_score_attr,
            base_score_attrs=args.base_score_attrs,
            i = args.i,
            j = args.j)