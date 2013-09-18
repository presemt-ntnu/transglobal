#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
evaluate Nearest Centroid models
"""

import logging

from sklearn.feature_selection import SelectFpr, chi2
from sklearn.pipeline import Pipeline

from tg.config import config
from tg.exps.support import ResultsStore, remove_exp_dir
from tg.utils import set_default_log
import tg.exps.experiment as ex
from tg.skl.selection import MinCountFilter, MaxFreqFilter
from tg.skl.centroid import NearestCentroidProb


log = logging.getLogger(__name__)


def nc_1(data_sets=config["eval"]["data_sets"],
         lang_pairs=(), n_graphs=None):
    name = "nc-1"
    remove_exp_dir(name)
    descriptor = [ 
        ("data", "S16"),
        ("source", "S8",  "source_lang"),
        ("target", "S8", "target_lang"),
        ("nist", "f", "scores.NIST"),
        ("bleu", "f", "scores.BLEU"),        
        ("exp_name", "S128"),       
        ("models_fname", "S256"),      
    ] 
    result_store = ResultsStore(descriptor, 
                                fname_prefix = "_" + name)
    classifier = Pipeline( [("MCF", MinCountFilter(5)),
                            ("MFF", MaxFreqFilter(0.1)),
                            ("CHI2", SelectFpr(chi2)),
                            ("CNC", NearestCentroidProb(metric="cosine"))
                            ])
    
    # 'data' cannot be expanded  implicitly through grid search
    # because _lang expansion depends on its value :-(
    for data in data_sets:
        exps = ex.single_exp(
            name=name,
            classifier=classifier,
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
        ##n_graphs=2,
    )
