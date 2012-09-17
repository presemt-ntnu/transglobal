#!/usr/bin/env python

"""
Compute "most frequent translation" (MFT) baseline scores
"""

import logging 
import cPickle 
import sys

import numpy as np

from tg.config import config
from tg.utils import set_default_log
from tg.exps.postproc import postprocess

log = logging.getLogger(__name__)


def most_frequent_translation(data_sets=config["eval"]["data_sets"],
                              lang_pairs=(),
                              draw=False, text=False):
    """
    compute most frequent translation (MFT) baseline scores on given data
    sets and language pairs (defaults to all available evaluation data)
    """
    results = []
    
    descriptor = {"names": ("data", "lang_pair", "NIST", "BLUE", "exp_name"), 
                  "formats": ("S16", "S8", "f4", "f4", "S64")}
    results = np.zeros(9999, dtype=descriptor)
    result_count = 0
    exp_name = "mft"
        
    for data_set in data_sets:
        for lang_pair in lang_pairs or config["eval"][data_set].keys():
            graphs_fname = config["eval"][data_set][lang_pair]["graphs_fname"]
            graph_list = cPickle.load(open(graphs_fname))
            
            nist_score, bleu_score = postprocess(exp_name, data_set,
                                                 lang_pair, graph_list, 
                                                 score_attr="freq_score",                                                  
                                                 sysid="most frequent translation",                                                  
                                                 draw=draw,
                                                 text=text)

            results[result_count] = ( data_set, lang_pair, 
                                      nist_score, bleu_score,
                                      exp_name ) 
            result_count += 1
            
    results = results[:result_count]    
    results_fname = "_mft_results"
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

most_frequent_translation()

# process Norwegain with output as text and graphs:
#most_frequent_translation(data_sets=("presemt-dev",),
#                          lang_pairs=("no-en", "no-de"),
#                          draw=True,
#                          text=True,
#                          )

