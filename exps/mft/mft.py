#!/usr/bin/env python

"""
Compute "most frequent translation" (MFT) baseline scores
"""

import logging    

from tg.config import config
from tg.utils import set_default_log
from tg.exps.preproc import preprocess
from tg.exps.postproc import postprocess


def most_frequent_translation(data_sets=("metis", "presemt-dev",
                                         "wmt08","wmt09", "wmt10", "wmt11"),
                              lang_pairs=None,
                              draw=False):
    """
    compute most frequent translation (MFT) baseline scores on given data
    sets and language pairs (defaults to all available evaluation data)
    """
    results = []
        
    for data_set in data_sets:
        for lang_pair in lang_pairs or config["eval"][data_set].keys():
            out_dir, exp_name, graph_list = preprocess(data_set, lang_pair)
            nist_score, bleu_score = postprocess(data_set, lang_pair,
                                                 out_dir, exp_name, 
                                                 graph_list, draw=draw)
            results.append((data_set, lang_pair, nist_score, bleu_score))
    
    print "DATA:\tLANG:\tNIST:\tBLEU"
    for result in results:
        print "{0}\t{1}\t{2}\t{3}".format(*result)
        
    return results

    

# for logging to stderr in utf-8 use:
set_default_log(level=logging.INFO)
#logging.getLogger("tg.annot").setLevel(logging.DEBUG)

most_frequent_translation()
#most_frequent_translation(data_sets=("presemt-dev",),
#                          lang_pairs=("no-en", "no-de"))

