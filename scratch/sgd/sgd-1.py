#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stochastic Gradient Descent with weighted classes
"""

import logging

from sklearn.linear_model import SGDClassifier

from tg.config import config
from tg.exps.support import ResultsStore, remove_exp_dir
from tg.utils import set_default_log
import tg.exps.experiment as ex
from tg.exps.support import grid_search



log = logging.getLogger(__name__)


# grid_search returns an *iterator* over classifiers
@grid_search    
def sgd_classifier(alpha=0.0001, 
                   class_weight=None, 
                   loss='log', 
                   n_iter=5, 
                   penalty='l2', 
                   ):
    return SGDClassifier(alpha=alpha,
                         class_weight=class_weight,
                         loss=loss,
                         n_iter=n_iter,
                         penalty=penalty,
                         shuffle=True)
    



def sgd_1(name = "sgd-1", 
          data_sets = ("presemt-dev",),
          lang=None,
          n_graphs=None):
    remove_exp_dir(name)
    descriptor = [ 
        ("data", "S16"),
        ("source", "S8",  "source_lang"),
        ("target", "S8", "target_lang"),
        ("loss", "S16", "classifier.loss"),
        ("penalty", "S16", "classifier.penalty"),
        ("alpha", "f", "classifier.alpha"),
        ("n_iter", "i", "classifier.n_iter"),
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
    classifiers = list(sgd_classifier(
        # These are the only two loss function that support log_proba.
        # The 'log' loss gives logistic regression, a probabilistic classifier.
        # 'modified_huber' is another smooth loss that brings tolerance to
        # outliers as well as probability estimates.
        _loss = ("log", "modified_huber"),
        _penalty = ('l2', 'l1', 'elasticnet'),
        _alpha = (0.0001, 0.001, 0.01, 0.1),
        _n_iter = (5,10,100),
    ))
    
    # 'data' cannot be expanded  implicitly through grid search
    # because _lang expansion depends on its value :-(
    for data in data_sets:
        exps = ex.single_exp(
            name=name,
            _classifier=classifiers,
            data=data,
            _lang=lang or config["eval"][data].keys(),
            write_text=ex.SKIP,
            draw_graphs=ex.SKIP,
            #build_models=sgd_build_models,
            n_graphs=n_graphs,
        )
        
        for ns in exps: 
            result_store.append(ns)
                
  
if __name__ == "__main__":
    set_default_log(log_fname="_sgd-1.log")
    sgd_1(
        n_graphs=2,
        lang=("de-en",),
    )
