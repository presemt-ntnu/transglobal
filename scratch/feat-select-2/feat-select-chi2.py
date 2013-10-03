#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CHI2 feature selection for range of alphas,
with Naive Bayes models with class priors from corpus,
with MFT vectors.

Note that for some combinations, no features at all are selected,
so there is no model for the word. As a result the number of "ignored"
in accuracy calculations may vary.
"""


import logging

from sklearn.feature_selection import SelectFpr, chi2
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

from tg.config import config
from tg.exps.support import ResultsStore, remove_exp_dir
from tg.utils import set_default_log
import tg.exps.experiment as ex
from tg.model import PriorModelBuilder
from tg.skl.selection import MinCountFilter, MaxFreqFilter
from tg.classcore import Vectorizer
from tg.exps.support import grid_search

log = logging.getLogger(__name__)


def nb_build_model(ns):
    """
    build NB models class priors from corpus counts
    """
    ns.models_fname = ns.fname_prefix + "_models.hdf5"           
    
    counts_fname = config["count"]["lemma"][ns.target_lang]["pkl_fname"]       
    model_builder = PriorModelBuilder(ns.get_data_generator(ns), 
                                      ns.models_fname,
                                      ns.classifier,
                                      counts_fname=counts_fname)
    model_builder.run()
    # clean up params
    delattr(ns, "ambig_map")


# grid_search returns an *iterator* over classifiers
@grid_search    
def nb_classifier(classifier=MultinomialNB,
                  min_count=5,
                  max_freq=0.1,
                  chi2_alpha=0.05,
                  nb_alpha=1.0):
    """
    Construct pipeline with feature selection and NB classifier
    """
    components = []
    
    if min_count:
        components.append(("MCF", MinCountFilter(min_count)))
    if max_freq:
        components.append(("MFF", MaxFreqFilter(max_freq)))
    if chi2_alpha:
        components.append(("CHI2", SelectFpr(chi2, alpha=chi2_alpha)))
    
    components.append(("MNB", classifier(alpha=nb_alpha)))
    return Pipeline(components)


def fs_1(data_sets=("metis, presemt-dev"), 
         n_graphs=None):
    name = "fs-1"
    remove_exp_dir(name)
    descriptor = [ 
        ("data", "S16"),
        ("source", "S8",  "source_lang"),
        ("target", "S8", "target_lang"),
        ("alpha", "f", "CHI2__alpha"),
        ("nist", "f", "scores.NIST"),
        ("bleu", "f", "scores.BLEU"),        
        ("exp_name", "S128"),   
    ] 
    result_store = ResultsStore(descriptor, 
                                fname_prefix = "_" + name)
    
    # tricky: 'classifiers' cannot be an iterator
    # because it is called many times during grid_search
    classifiers = list(nb_classifier(
        min_count=5,
        max_freq=None,
        _chi2_alpha=[0.5, 0.1, 0.05, 0.01, 0.005, 0.0001, 0.00001]))
    
    vectorizer=Vectorizer(score_attr="freq_score")
    
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
            build_models=nb_build_model,
            vectorizer=vectorizer,
            thrash_models=ex.thrash_models,
            n_graphs=n_graphs,
        )
        
        for ns in exps: 
            result_store.append(ns)
                
  
if __name__ == "__main__":
    set_default_log(log_fname="_fs-1.log")
    fs_1(data_sets=(
        "metis", 
        "presemt-dev"
        ),
         #n_graphs=2
         )
