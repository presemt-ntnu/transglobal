#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import numpy as np
import asciitable as at

from tg.utils import grid_search, set_default_log
import tg.exps.experiment as ex

from nb import nb_classifier, nb_build_models

log = logging.getLogger(__name__)


def nb_1():
    name = "nb-1"
        
    descriptor = [ ("prior", "b"),
                   ("attr", "S16"),
                   ("data", "S16"),
                   ("lang", "S8"),
                   ("nist", "f"),
                   ("bleu", "f"),
                   ("name", "S128"),
                   ("models_fname", "S256"),
                    ] 
    results = np.zeros(9999, dtype=descriptor)
    results_fname = "_" + name + "_results.npy"
    results_txt_fname = "_" + name + "_results.txt"
    
    classifiers = grid_search(nb_classifier, 
                              _min_count=(None, 5, 10))
    
    exps = grid_search(ex.single_exp,
                       name=name,
                       _classifier=classifiers,
                       _data=("metis",
                              #"presemt-dev",
                              ),
                       _lang=("de-en",
                              #"en-de",
                              ),
                       write_text=ex.SKIP,
                       draw_graphs=ex.SKIP,
                       _class_priors=(True, False),
                       #save_graphs=True,
                       #save_models=True,
                       #text=True,
                       #write_diff=ex.write_diff,
                       draw=ex.SKIP,
                       n_graphs=2,
                       build_models=nb_build_models,
                       )
    
    for exp_count, ns in enumerate(exps): 
        results[exp_count]["prior"] = ns.class_priors
        results[exp_count]["data"] = ns.data
        results[exp_count]["lang"] = ns.lang
        results[exp_count]["nist"] = ns.scores.NIST
        results[exp_count]["bleu"] = ns.scores.BLEU
        results[exp_count]["name"] = ns.exp_name
        results[exp_count]["models_fname"] = ns.models_fname
        # save each intermediary result
        log.info("saving pickled results to " + results_fname)
        np.save(results_fname, results[:exp_count + 1]) 
        log.info("saving results table to " + results_txt_fname)
        at.write(results[:exp_count + 1], results_txt_fname, 
                 Writer=at.FixedWidthTwoLine, delimiter_pad=" ")
                

                
if __name__ == "__main__":
    set_default_log()
    nb_1()
