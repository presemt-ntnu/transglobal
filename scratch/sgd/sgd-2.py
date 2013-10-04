#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reuse models from sgd-1.py to run exps with different vectorizers
"""

import logging

import numpy as np

from tg.config import config
from tg.exps.support import ResultsStore, remove_exp_dir
from tg.utils import set_default_log
import tg.exps.experiment as ex
from tg.classcore import Vectorizer
from tg.exps.support import grid_search


log = logging.getLogger(__name__)



@grid_search
def vectorizer(score_attr=None, min_score=None):
    return Vectorizer(score_attr=score_attr, min_score=min_score)




def sgd_2(name = "sgd-2", n_graphs=None):
    remove_exp_dir(name)
    descriptor = [ 
        ("data", "S16"),
        ("source", "S8",  "source_lang"),
        ("target", "S8", "target_lang"),
        ("vect_score_attr", "S16", "vectorizer.score_attr"),
        ("nist", "f", "scores.NIST"),
        ("bleu", "f", "scores.BLEU"),  
        ("correct", "i", "accuracy.correct"),
        ("incorrect", "i", "accuracy.incorrect"),
        ("ignored", "i", "accuracy.ignored"),
        ("accuracy", "f", "accuracy.score"),
        ("exp_name", "S128"),        
        ("models_fname", "S256"),
    ] 
    result_store = ResultsStore(descriptor, 
                                fname_prefix = "_" + name)
    vectorizers=list(vectorizer(
        _score_attr=(None, "freq_score", "dup_score")))
    sgd_1_results = np.load("_sgd-1.npy")
    
    for record in sgd_1_results:
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
    set_default_log(log_fname="_sgd-2.log")
    sgd_2(
        #name="test",
        #n_graphs=2,
    )
