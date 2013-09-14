#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from tg.exps.support import ResultsStore, remove_exp_dir
from tg.utils import set_default_log
import tg.exps.experiment as ex

from nb import nb_classifier, nb_build_models

log = logging.getLogger(__name__)


def nb_1():
    name = "nb-1"
    remove_exp_dir(name)
    descriptor = [ ("prior", "b", "class_priors"),
                   ("data", "S16"),
                   ("lang", "S8"),
                   ("min_count", "f", 
                    "classifier.get_params().get('MCF__min_count')"),
                   ("nist", "f", "scores.NIST"),
                   ("bleu", "f", "scores.BLEU"),
                   ("exp_name", "S128"),
                   ("models_fname", "S256"),
                    ] 
    result_store = ResultsStore(descriptor, 
                                fname_prefix = "_" + name)
    
    classifiers = nb_classifier(
        _min_count=(None, 5, 10))
    
    exps = ex.single_exp(
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
    
    for ns in exps: 
        result_store.append(ns)
                

                
if __name__ == "__main__":
    set_default_log()
    nb_1()
