#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
evaluate diffeent Naive Bayes models on Presemt evaluation data
"""

# TODO:
# - log to file


import logging
import os
from os.path import join

import numpy as np

from sklearn.naive_bayes import MultinomialNB, BernoulliNB

from tg.config import config
from tg.utils import set_default_log

from nb_exp import prepare, score_model
from nb_model import make_models, extract_source_lempos_subset




def prepare_graphs():
    """
    prepare annotated graphs with frequency scoring
    """
    prepare("en-de")
    prepare("de-en")
    

def prepare_samples():
    """
    prepare filtered and extended samples
    """
    # TODO
    # for now, just run in the cl scripts:
    # convert-to_hdf5.py
    # filter_vocab.py
    # extend.py
    pass

def _score_model(*args, **kwargs):
    return 99.9, 0.99
    
    
def run_all():
    language_pairs = "en-de", "de-en"
    tab_fnames = ( join(config["private_data_dir"], "corpmod/en/de/en-de_ambig.tab"),
                   join(config["private_data_dir"], "corpmod/de/en/de-en_ambig.tab") )
    extended_vectors = True, #False
    classifier_types = MultinomialNB, BernoulliNB
    alpha_values = 1.0, 0.1, 0.01, np.finfo(np.double).eps
    corpus_prior_values = False, True
    # the fit_prior parameters seems to make absolutely no difference
    # fit_prior_values = True, False
    
    descriptor = {'names': ('lang_pair', 'classifier', 'alpha', 
                            'corpus_priors', 'extended', 'NIST', 'BLUE'), 
                  'formats': ('S8', 'S64', 'f4', 'b', 'b', 'f4', 'f4')}
    results = np.zeros(9999, dtype=descriptor)
    exp_count = 0
    
    for lang_pair, tab_fname in zip(language_pairs, tab_fnames):
        graphs_pkl_fname = "prep/{}_graphs.pkl".format(lang_pair)
        lempos_subset = extract_source_lempos_subset(graphs_pkl_fname)
        target_lang = lang_pair.split("-")[1]
        
        for extended in extended_vectors:
            if extended:
                samp_fname = target_lang + "_samples_subset_filtered_extended.hdf5"
            else:
                samp_fname = target_lang + "_samples_subset_filtered.hdf5"
                
            for corpus_prior in corpus_prior_values:
                if corpus_prior:
                    counts_pkl_fname = config["count"]["lemma"][target_lang]["pkl_fname"]
                else:
                    counts_pkl_fname = None
            
                for classifier_class in classifier_types:
                    for alpha in alpha_values:
                        results[exp_count] = ( lang_pair, 
                                               classifier_class.__name__,
                                               alpha,
                                               corpus_prior,
                                               extended,
                                               0,
                                               0
                                               )
                        exp_dir = "exp_" + "_".join("{}={}".format(var, value) 
                                                    for var, value in zip(results.dtype.names, 
                                                                          results[exp_count])[:-2])
                        if not os.path.exists(exp_dir):
                            os.makedirs(exp_dir)
                        classifier = classifier_class(alpha=alpha)
                        models_fname = join(exp_dir, "nb_models.hdf5")
                        make_models(tab_fname, samp_fname, models_fname, classifier,
                                    source_lempos_subset=lempos_subset, 
                                    counts_pkl_fname=counts_pkl_fname) 
                        nist, blue = score_model(lang_pair, exp_dir, draw=False)
                        results[exp_count]["NIST"] = nist 
                        results[exp_count]["BLUE"] = blue
                        exp_count += 1
            
    results = results[:exp_count]
    print results
    results.dump("results.pkl")
    
    


# for logging to stderr in utf-8 use:
set_default_log(level=logging.INFO)

#import logging
#logging.getLogger("tg.nbscore").setLevel(logging.DEBUG)    

prepare_graphs()

##run_all()    