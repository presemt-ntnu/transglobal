#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Report statistics on translation ambiguity in dictionary
"""


from argparse import ArgumentParser
from cPickle import load
from os.path import basename
import logging as log

log = log.getLogger(basename(__file__))

from tg.config import config
from tg.transdict import ambig_dist_report
from tg.utils import set_default_log


parser = ArgumentParser(description=__doc__)    

parser.add_argument(
    "-l", "--lang-pairs",
    nargs="+",
    metavar="SL-TL",
    default=config["dict"].keys(),
    help="language pair(s); default is:" + 
    ", ".join(config["dict"].keys()))

parser.add_argument(
    "-e", "--entry",
    choices=["lempos", "lemma"],
    default="lempos",
    help="count lemma + POS entries or lemma only entries; "
    "default is lempos")

parser.add_argument(
    "--no-single-word",
    dest="with_single_word",
    default=True,
    action="store_false",
    help="count single word entries (default is True)")

parser.add_argument(
    "--with-multi-word",
    default=False,
    action="store_true",
    help="cont multi word entries (default is False)")

parser.add_argument(
    "-v", "--verbose",
    action="store_true")

args = parser.parse_args()

if args.verbose:
    set_default_log()
            
            
ambig_dist_report(lang_pairs=args.lang_pairs, 
                  entry=args.entry,
                  with_single_word=args.with_single_word,
                  with_multi_word=args.with_multi_word)

        
    
    
    

