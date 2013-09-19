#!/usr/bin/env python

"""
Compute upper and lower bounds on scores
"""

import logging 
import cPickle 
import sys

from tg.config import config
from tg.exps.support import ResultsStore, remove_exp_dir
from tg.utils import set_default_log
import tg.exps.experiment as ex

log = logging.getLogger(__name__)


def bounds(data_sets=config["eval"]["data_sets"], lang_pairs=()): 
    """
    Compute upper and lower bounds on scores. 
    
    The baseline that serves as the lower bound is the Most Frequent
    Translation (MFT) score, which is obtained by choosing the translation
    with te highest frequency in the target language corpus.
    
    Upper bound is the Approximated Maximum (AM) score, which is obtained by
    choosing the translation that occurs most often in the reference
    translations(s) of the sentence. 
    
    Probility scores for lempos translations are already in the preprocessed
    graphs. This function just computes the resulting NIST and BLEU scores.
    """
    name = "bounds"
    remove_exp_dir(name)
    descriptor = [ 
        ("data", "S16"),
        ("source", "S8",  "source_lang"),
        ("target", "S8", "target_lang"),
        ("score_attr", "S16", "score_attr"),
        ("nist", "f", "scores.NIST"),
        ("bleu", "f", "scores.BLEU"),        
        ("exp_name", "S128"),        
    ] 
    result_store = ResultsStore(descriptor, 
                                fname_prefix = "_" + name)
    
    for data in data_sets:
        exps = ex.single_exp(
            name=name,
            classifier=None,
            data=data,
            _lang=lang_pairs or config["eval"][data].keys(),
            _score_attr=("freq_score","dup_score", "mup_score"),
            build=ex.SKIP,
            compute_classifier_score=ex.SKIP,
            write_text=ex.SKIP,
            write_diff=ex.SKIP,
            draw_graphs=ex.SKIP)
        
        for ns in exps: 
            result_store.append(ns)
                
  
if __name__ == "__main__":
    set_default_log(log_fname="_bounds.log")
    bounds()

    
    
