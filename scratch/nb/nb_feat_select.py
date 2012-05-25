#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
evaluate different feature selection methods with Naive Bayes models on Presemt evaluation data
"""

# TODO:
# - log to file


import logging
import os
from os.path import join

import numpy as np

from scipy import stats

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_selection import SelectKBest, chi2

from tg.config import config
from tg.utils import set_default_log

from nb_exp import prepare, score_model

from model import NBModelBuilder


# from sklearn.feature_selection import SelectKBest, SelectPercentile

#class _SelectPercentile(SelectPercentile):
    #"""
    #Fix for NaN issue
    
    #The problem is that stats.scoreatpercentile may return np.nan. In that
    #case, the support mask is empty, because nothing is equal or smaller than
    #nan... It seems reasonable to return all features then. This can be accomplised 
    #"""
    
    #def _get_support_mask(self):
        #percentile = self.percentile
        #assert percentile<=100, ValueError('percentile should be \
                            #between 0 and 100 (%f given)' %(percentile))
        ## Cater for Nans
        #if percentile == 100:
            #return np.ones(len(self._pvalues), dtype=np.bool)
        #elif percentile == 0:
            #return np.zeros(len(self._pvalues), dtype=np.bool)
        #alpha = stats.scoreatpercentile(self._pvalues, percentile)
        
        #if not np.isnan(alpha):
            #return (self._pvalues <= alpha)
        #else:
            #return np.asarray(self._pvalues, dtype="bool")


def run_all():
    language_pairs =  "de-en", "en-de"
    tab_fnames = ( join(config["private_data_dir"], "corpmod/de/en/de-en_ambig.tab"),
                   join(config["private_data_dir"], "corpmod/en/de/en-de_ambig.tab"))
    
    descriptor = {'names': ('lang_pair', 'k', 'NIST', 'BLUE'), 
                  'formats': ('S8', 'i4', 'f4', 'f4')}
    #percentiles = [0.25, 0.5, 1, 2.5, 5, 10, 20, 25, 30, 40, 50, 60, 70, 80, 90]
    #p_values = [0.1, 0.05, 0.01, 0.005, 0.001]
    #p_values = [0.0005, 0.0001, 0.00005, 0.00001]
    #p_values = [0.00005, 0.00001, 0.000005, 0.000001, 0.0000005, 0.0000001]
    #p_values = [0.00000005, 0.00000001, 0.000000005, 0.000000001, 0.0000000005, 0.000000001]
    #k_values = [1,2,3,4,5,10,25,50,100,250,500,1000]
    k_values = [150,200,300,350,400,450]
    results = np.zeros(9999, dtype=descriptor)
    exp_count = 0
    
    
    for lang_pair, tab_fname in zip(language_pairs, tab_fnames):
        target_lang = lang_pair.split("-")[1]
        samp_fname = target_lang + "_samples_subset_filtered.hdf5"
        graphs_pkl_fname = "prep/{}_graphs.pkl".format(lang_pair)
        target_lang = lang_pair.split("-")[1]
        counts_pkl_fname = config["count"]["lemma"][target_lang]["pkl_fname"]
        
        for k in k_values:
            results[exp_count] = ( lang_pair, 
                                   k,
                                   0,
                                   0
                                   )
            exp_dir = "exp_" + "_".join("{}={}".format(var, value) 
                                        for var, value in zip(results.dtype.names, 
                                                              results[exp_count])[:-2])
            if not os.path.exists(exp_dir):
                os.makedirs(exp_dir)
            #else:
            #    continue
            
            classifier = MultinomialNB()
            models_fname = join(exp_dir, "nb_models.hdf5")
            
            builder = NBModelBuilder(tab_fname, samp_fname, models_fname,
                                     classifier, graphs_pkl_fname=graphs_pkl_fname,
                                     counts_pkl_fname=counts_pkl_fname,
                                     feat_selector=SelectKBest(chi2, k))
            builder.run()
            nist, blue = score_model(lang_pair, exp_dir, draw=False)
            results[exp_count]["NIST"] = nist 
            results[exp_count]["BLUE"] = blue
            
            exp_count += 1
            
    results = results[:exp_count]
    print results
    results.dump("nb_feat_select_bst_results_2.pkl")
    
    



# for logging to stderr in utf-8 use:
set_default_log(level=logging.INFO)

import logging
logging.getLogger("model").setLevel(logging.DEBUG)    

run_all()    