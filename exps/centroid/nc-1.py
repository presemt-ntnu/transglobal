#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
evaluate Nearest Centroid models
"""

import logging

from sklearn.feature_selection import SelectFpr, chi2
from sklearn.pipeline import Pipeline

from tg.config import config
from tg.exps.support import ResultsStore, remove_exp_dir, grid_search
from tg.utils import set_default_log
import tg.exps.experiment as ex
from tg.skl.selection import MinCountFilter, MaxFreqFilter
from tg.skl.centroid import NearestCentroidProb



log = logging.getLogger(__name__)



# grid_search returns an *iterator* over classifiers
@grid_search    
def nc_classifier(min_count=5,
                  max_freq=0.05,
                  chi2_alpha=None,
                  metric="cosine"):
    """
    Construct pipeline with feature selection and NC classifier
    """
    components = []
    
    if min_count:
        components.append(("MCF", MinCountFilter(min_count)))
    if max_freq:
        components.append(("MFF", MaxFreqFilter(max_freq)))
    if chi2_alpha:
        components.append(("CHI2", SelectFpr(chi2, alpha=chi2_alpha)))
    
    classifier = NearestCentroidProb(metric=metric)
    components.append(("NCC", classifier))
    return Pipeline(components)


def nc_1(data_sets=config["eval"]["data_sets"],
         lang_pairs=(), n_graphs=None,
         name = "nc-1"):
    remove_exp_dir(name)
    descriptor = [ 
        ("data", "S16"),
        ("source", "S8",  "source_lang"),
        ("target", "S8", "target_lang"),
        ("metric", "S16", "NCC__metric"),
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
    classifiers = list(nc_classifier(
        # Contrary to docs, l1 distance (manhattan) does NOT support sparse 
        _metric=("cosine", "euclidean")))
    
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
            n_graphs=n_graphs,
        )
        
        for ns in exps: 
            result_store.append(ns)
                
  
if __name__ == "__main__":
    set_default_log(log_fname="_nc_1.log")
    nc_1(
        data_sets = ("metis","presemt-dev"),
        #n_graphs=2,
    )
