#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logistic Regression (MaxEnt)

This is very slow. Better use SGDClassifier wth loss="log"
"""

import logging

from sklearn.linear_model import LogisticRegression

from tg.config import config
from tg.exps.support import ResultsStore, remove_exp_dir
from tg.utils import set_default_log
import tg.exps.experiment as ex
from tg.exps.support import grid_search



log = logging.getLogger(__name__)


# grid_search returns an *iterator* over classifiers
@grid_search    
def lr_classifier():
    return LogisticRegression()
    



def lr_1(name = "lr-1", 
          data_sets = ("presemt-dev",),
          n_graphs=None):
    remove_exp_dir(name)
    descriptor = [ 
        ("data", "S16"),
        ("source", "S8",  "source_lang"),
        ("target", "S8", "target_lang"),
        ("loss", "S16", "classifier.loss"),
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
    classifiers = list(lr_classifier(
        ))
    
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
            #build_models=lr_build_models,
            n_graphs=n_graphs,
        )
        
        for ns in exps: 
            result_store.append(ns)
                
  
if __name__ == "__main__":
    set_default_log(log_fname="_lr-1.log")
    lr_1(
        n_graphs=1
    )
