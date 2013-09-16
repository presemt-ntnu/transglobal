#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reuse models from nb-1.py to run exps with different vectorizers
"""

import logging

import numpy as np

from tg.config import config
from tg.exps.support import ResultsStore, remove_exp_dir
from tg.utils import set_default_log
import tg.exps.experiment as ex

from nb import vectorizer


log = logging.getLogger(__name__)


def nb_2():
    ##name = "test"
    name = "nb-2"
    remove_exp_dir(name)
    descriptor = [ 
        ("data", "S16"),
        ("source", "S8",  "source_lang"),
        ("target", "S8", "target_lang"),
        ("prior", "b", "class_priors"),
        ("vect_score_attr", "S16", "vectorizer.score_attr"),
        ("nist", "f", "scores.NIST"),
        ("bleu", "f", "scores.BLEU"),        
        ("exp_name", "S128"),        
        ("models_fname", "S256"),
    ] 
    result_store = ResultsStore(descriptor, 
                                fname_prefix = "_" + name)
    vectorizers=list(vectorizer(_score_attr=(None, "freq_score")))
    nb_1_results = np.load("_nb-1.npy")
    
    for record in nb_1_results:
        exps = ex.single_exp(
                    name=name,
                    data=record["data"],
                    lang=record["source"] + "-" + record["target"],
                    classifier=None,
                    class_priors=record["prior"],
                    write_text=ex.SKIP,
                    draw_graphs=ex.SKIP,
                    build_models=ex.SKIP,
                    trash_models=ex.SKIP,
                    models_fname=record["models_fname"],
                    _vectorizer=vectorizers,
                    ##n_graphs=2,
                )    
        
        for ns in exps: 
            result_store.append(ns)
                
  
if __name__ == "__main__":
    set_default_log()
    nb_2()
