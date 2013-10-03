#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stochastic Gradient Descent

TODO: class weights
"""

import logging

from sklearn.linear_model import SGDClassifier

from tg.config import config
from tg.exps.support import ResultsStore, remove_exp_dir
from tg.utils import set_default_log
import tg.exps.experiment as ex
from tg.exps.support import grid_search



log = logging.getLogger(__name__)


    

def sgd_1(name = "sgd-1", 
          data_sets = ("metis", "presemt-dev",),
          lang=None,
          n_graphs=None,
          n_jobs=1):
    remove_exp_dir(name)
    descriptor = [ 
        ("data", "S16"),
        ("source", "S8",  "source_lang"),
        ("target", "S8", "target_lang"),
        #("loss", "S16", "classifier.loss"),
        #("penalty", "S16", "classifier.penalty"),
        #("alpha", "f", "classifier.alpha"),
        #("n_iter", "i", "classifier.n_iter"),
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
    # best setting found in sgd-cv exps
    classifier = SGDClassifier(loss = "log",
                               penalty = "l2",
                               alpha = 0.001,
                               n_iter = 5,
                               shuffle=True,
                               n_jobs=n_jobs)
    
    # 'data' cannot be expanded  implicitly through grid search
    # because _lang expansion depends on its value :-(
    for data in data_sets:
        exps = ex.single_exp(
            name=name,
            classifier=classifier,
            data=data,
            _lang=lang or config["eval"][data].keys(),
            write_text=ex.SKIP,
            draw_graphs=ex.SKIP,
            n_graphs=n_graphs,
            # *** input to SGDClassifier must be shuffled! ***
            shuffle=True,
        )
        
        for ns in exps: 
            result_store.append(ns)
                
  
if __name__ == "__main__":
    set_default_log(log_fname="_sgd-1.log")
    sgd_1(
        n_graphs=2,
        #lang=("de-en",),
        #n_jobs=10
    )
