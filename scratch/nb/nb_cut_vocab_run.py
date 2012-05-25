#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
evaluate Naive Bayes models on Presemt evaluation data
with different frequency cut-offs for vocabulary
"""


# TODO:
# - log to file

import cPickle
import logging
import os
from os.path import join

import numpy as np

import h5py

from sklearn.naive_bayes import MultinomialNB, BernoulliNB

from tg.config import config
from tg.utils import set_default_log

from nb_exp import score_model
from nb_model import make_models, extract_source_lempos_subset



def cut_vocab(samp_fname, counts_pkl_fname, bins=10):
    hdfile = h5py.File(samp_fname)
    counts_dict = cPickle.load(open(counts_pkl_fname))
    vocab_counts = [ counts_dict[lemma.decode("utf-8")]
                     for lemma in  hdfile["vocab"] ]
    hdfile.close()
    
    total = sum(vocab_counts)
    bin_size = total / bins
    cum_sum = 0
    indices = []
    
    for i, count in enumerate(vocab_counts):
        cum_sum += count
        if cum_sum > bin_size:
            indices.append(i+1)
            cum_sum = 0
            
    # debug code
    for k in range(len(indices) + 1):
        if k > 0:
            i = indices[k-1]
        else:
            i = None
            
        try:
            j = indices[k]
        except IndexError:
            j = None
            
        bin_total = sum(vocab_counts[i:j])

        print "{0:>8}{1:>8}{2:>16d}{3:>16d}".format(i, j, bin_total, bin_total - bin_size) 
        
    return [ (i,j) 
             for i in [0] + indices
             for j in indices + [len(vocab_counts)]
             if vocab_counts[i:j] ]



def run_all():
    language_pairs = "en-de", "de-en"
    tab_fnames = ( join(config["private_data_dir"], "corpmod/en/de/en-de_ambig.tab"),
                   join(config["private_data_dir"], "corpmod/de/en/de-en_ambig.tab") )
    
    descriptor = {'names': ('lang_pair', 'vocab_i', 'vocab_j', 'NIST', 'BLUE'), 
                  'formats': ('S8','i4', 'i4', 'f4', 'f4')}
    results = np.zeros(9999, dtype=descriptor)
    exp_count = 0
    
    for lang_pair, tab_fname in zip(language_pairs, tab_fnames):
        target_lang = lang_pair.split("-")[1]
        samp_fname = target_lang + "_samples_subset_filtered.hdf5"
        graphs_pkl_fname = "prep/{}_graphs.pkl".format(lang_pair)
        lempos_subset = extract_source_lempos_subset(graphs_pkl_fname)
        target_lang = lang_pair.split("-")[1]
        counts_pkl_fname = config["count"]["lemma"][target_lang]["pkl_fname"]
        
        for vocab_i, vocab_j in cut_vocab(samp_fname, counts_pkl_fname): 
            results[exp_count] = ( lang_pair, 
                                   vocab_i,
                                   vocab_j,
                                   0,
                                   0
                                   )
            exp_dir = "exp_" + "_".join("{}={}".format(var, value) 
                                        for var, value in zip(results.dtype.names, 
                                                              results[exp_count])[:-2])
            if not os.path.exists(exp_dir):
                os.makedirs(exp_dir)
            classifier = MultinomialNB()
            models_fname = join(exp_dir, "nb_models.hdf5")
            make_models(tab_fname, samp_fname, models_fname, classifier,
                        source_lempos_subset=lempos_subset, 
                        counts_pkl_fname=counts_pkl_fname,
                        vocab_i=vocab_i, vocab_j=vocab_j) 
            nist, blue = score_model(lang_pair, exp_dir, draw=False)
            results[exp_count]["NIST"] = nist 
            results[exp_count]["BLUE"] = blue
            exp_count += 1
            
    results = results[:exp_count]
    print results
    results.dump("nb_cut_vocab_results.pkl")
    
    


# for logging to stderr in utf-8 use:
set_default_log(level=logging.INFO)

#import logging
logging.getLogger("__main__").setLevel(logging.DEBUG)    

run_all()    