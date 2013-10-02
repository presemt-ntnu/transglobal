#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stochastic Gradient Descent
"""

import logging

from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectFpr, chi2
from sklearn.pipeline import Pipeline

from tg.config import config
from tg.exps.support import ResultsStore, remove_exp_dir
from tg.utils import set_default_log
import tg.exps.experiment as ex
from tg.exps.support import grid_search
from tg.skl.selection import MinCountFilter, MaxFreqFilter


log = logging.getLogger(__name__)


# grid_search returns an *iterator* over classifiers
@grid_search    
def sgd_classifier(alpha=0.0001, 
                   class_weight=None, 
                   loss='log', 
                   n_iter=5, 
                   penalty='l2',
                   n_jobs=1,
                   min_count=5,
                   max_freq=0.05,
                   chi2_alpha=None):
        """
        Construct pipeline with feature selection and SGD classifier
        """
        components = []

        if min_count:
            components.append(("MCF", MinCountFilter(min_count)))
        if max_freq:
            components.append(("MFF", MaxFreqFilter(max_freq)))
        if chi2_alpha:
            components.append(("CHI2", SelectFpr(chi2, alpha=chi2_alpha)))

        classifier = SGDClassifier(alpha=alpha,
                                   class_weight=class_weight,
                                   loss=loss,
                                   n_iter=n_iter,
                                   penalty=penalty,
                                   shuffle=True,
                                   n_jobs=n_jobs)
        components.append(("SGD", classifier))
        return Pipeline(components)


def sgd_1(name = "sgd-1", 
          data_sets = ("presemt-dev",),
          lang=None,
          n_jobs=1,
          n_graphs=None):
    remove_exp_dir(name)
    descriptor = [ 
        ("data", "S16"),
        ("source", "S8",  "source_lang"),
        ("target", "S8", "target_lang"),
        ("loss", "S16", "SGD__loss"),
        ("penalty", "S16", "SGD__penalty"),
        ("alpha", "f", "SGD__alpha"),
        ("n_iter", "i", "SGD__n_iter"),
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
        _penalty = ('l2', 'l1'), # elasticnet
        _alpha = (0.00001, 0.0001, 0.001),
        _n_iter = (5,10),
        n_jobs=n_jobs,
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
        #n_graphs=2,
        n_jobs=10,
        #lang=("de-en",),
    )
