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

from nb import nb_classifier, nb_build_models

log = logging.getLogger(__name__)


def nb_1():
    ##name = "test"
    name = "nb-1"
    remove_exp_dir(name)
    descriptor = [ 
        ("data", "S16"),
        ("source", "S8",  "source_lang"),
        ("target", "S8", "target_lang"),
        ("prior", "b", "class_priors"),
        ("nist", "f", "scores.NIST"),
        ("bleu", "f", "scores.BLEU"),        
        ("exp_name", "S128"),        
        ("models_fname", "S256"),
    ] 
    result_store = ResultsStore(descriptor, 
                                fname_prefix = "_" + name)
    data_sets = "metis", "presemt-dev"
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
            _lang=config["eval"][data].keys(),
            #_lang=("de-en",),
            write_text=ex.SKIP,
            draw_graphs=ex.SKIP,
            _class_priors=(True, False),
            build_models=nb_build_models,
            ##n_graphs=2,
        )
        
        for ns in exps: 
            result_store.append(ns)
                
  
if __name__ == "__main__":
    set_default_log(log_fname="_nb-1.log")
    nb_1()
