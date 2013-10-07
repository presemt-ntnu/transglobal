#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Naive Bayes models with/without class priors from corpus
"""


import logging

from tg.config import config
from tg.exps.support import ResultsStore, remove_exp_dir
from tg.utils import set_default_log
import tg.exps.experiment as ex

from nb import nb_classifier
log = logging.getLogger(__name__)


def nb_1(name = "nb-1",
         data_sets=("metis", "presemt-dev"),
         lang_pairs=None,
         n_graphs=None):
    remove_exp_dir(name)
    descriptor = [ 
        ("data", "S16"),
        ("source", "S8",  "source_lang"),
        ("target", "S8", "target_lang"),
        ("class_weighting", "b"),
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
    # tricky: 'classifiers' cannot be an iterator
    # because it is called many times during grid_search
    classifiers = list(nb_classifier())
    
    # 'data' cannot be expanded  implicitly through grid search
    # because _lang expansion depends on its value :-(
    for data in data_sets:
        exps = ex.single_exp(
            name=name,
            _classifier=classifiers,
            data=data,
            _lang=lang_pairs or config["eval"][data].keys(),
            write_text=ex.SKIP,
            draw_graphs=ex.SKIP,
            _class_weighting=(True, False),
            n_graphs=n_graphs,
        )
        
        for ns in exps: 
            result_store.append(ns)
                
  
if __name__ == "__main__":
    set_default_log(log_fname="_nb-1.log")
    nb_1(
        #name="ff",
        #n_graphs=2
    )
