#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reuse models from nc-1.py to run exps with different vectorizers
"""

import logging

import numpy as np

from tg.config import config
from tg.classcore import Vectorizer
from tg.exps.support import ResultsStore, remove_exp_dir
from tg.utils import set_default_log
import tg.exps.experiment as ex


log = logging.getLogger(__name__)


def nc_2(name = "nc-2", n_graphs=None):
    remove_exp_dir(name)
    descriptor = [ 
        ("data", "S16"),
        ("source", "S8",  "source_lang"),
        ("target", "S8", "target_lang"),
        ("vect_score_attr", "S16", "vectorizer.score_attr"),
        ("nist", "f", "scores.NIST"),
        ("bleu", "f", "scores.BLEU"),        
        ("exp_name", "S128"),        
        ("models_fname", "S256"),
    ] 
    result_store = ResultsStore(descriptor, 
                                fname_prefix = "_" + name)
    vectorizers= [Vectorizer(score_attr=score_attr) 
                  for score_attr in (None, "freq_score", "dup_score")]
    nc_1_results = np.load("_nc-1.npy")
    
    for record in nc_1_results:
        exps = ex.single_exp(
                    name=name,
                    data=record["data"],
                    lang=record["source"] + "-" + record["target"],
                    classifier=None,
                    write_text=ex.SKIP,
                    draw_graphs=ex.SKIP,
                    build_models=ex.SKIP,
                    trash_models=ex.SKIP,
                    models_fname=record["models_fname"],
                    _vectorizer=vectorizers,
                    n_graphs=n_graphs,
                )    
        
        for ns in exps: 
            result_store.append(ns)
                
  
if __name__ == "__main__":
    set_default_log(log_fname="_nc-2.log")
    nc_2(
        ## n_graphs=2,
    )
