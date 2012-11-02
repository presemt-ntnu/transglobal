#!/usr/bin/env python

"""
Compute approximated maximum scores
"""

import logging 
import cPickle 
import sys

import numpy as np

from tg.config import config
from tg.utils import set_default_log
from tg.exps.postproc import postprocess
from tg.maxscore import MaxScore

log = logging.getLogger(__name__)


def approximate_max_score(data_sets=config["eval"]["data_sets"],
                          lang_pairs=(),
                          draw=False, text=False):
    """
    Approximation of the maximal scoring obtainable.
    
    For each segment (sentence), we look at its reference translation(s) and
    count the number of times that a target lemma occurs. Next, when scoring
    translations candidates, we pick the translations with the highest count
    in the reference translation.
    """
    results = []
    
    descriptor = {"names": ("data", "lang_pair", "NIST", "BLUE", "exp_name"), 
                  "formats": ("S16", "S8", "f4", "f4", "S64")}
    results = np.zeros(9999, dtype=descriptor)
    result_count = 0
    exp_name = "ams"
        
    for data_set in data_sets:
        for lang_pair in lang_pairs or config["eval"][data_set].keys():
            graphs_fname = config["eval"][data_set][lang_pair]["graphs_fname"]
            graph_list = cPickle.load(open(graphs_fname))
            
            lemma_ref_fname = \
            config["eval"][data_set][lang_pair]["lemma_ref_fname"]
            maxscore = MaxScore(lemma_ref_fname)
            maxscore(graph_list)
            
            nist_score, bleu_score = postprocess(exp_name, data_set,
                                                 lang_pair, graph_list, 
                                                 best_score_attr="max_score",
                                                 draw=draw,
                                                 text=text)

            results[result_count] = ( data_set, lang_pair, 
                                      nist_score, bleu_score,
                                      exp_name ) 
            result_count += 1
            
    results = results[:result_count]    
    results_fname = "_ams_results"
    log.info("saving results to " + results_fname)
    np.save(results_fname, results)
    
    print "%-16s\t%-8s\t%8s\t%8s\t%s" % ("DATA:", "LANG:", 
                                         "NIST:", "BLUE:", "NAME:")
    for row in results:
        print "%-16s\t%-8s\t%8.4f\t%8.4f\t%s" % tuple(row)
        
    return results

    

# for logging to stderr in utf-8 use:
set_default_log(level=logging.INFO)
#logging.getLogger("tg.annot").setLevel(logging.DEBUG)

approximate_max_score()

# process Norwegain with output as text and graphs:
#most_frequent_translation(data_sets=("presemt-dev",),
#                          lang_pairs=("no-en", "no-de"),
#                          draw=True,
#                          text=True,
#                          )

