#!/usr/bin/env python

"""
Attempt at cross-validated recursive feature elimination.

Seems not to work with Naive Bayes.
LinearSVC is too slow.
With SGD, everything with less features seems to score lower.
Puzzling results... there must be someting wrong here.
"""


import codecs
import sys

import numpy as np
import pylab as pl
import h5py

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.feature_selection.rfe import RFECV
from sklearn.metrics import f1_score, precision_recall_fscore_support, zero_one_loss

from tg.config import config
from tg.ambig import AmbiguityMap
from tg.sample import DataSetGenerator


def score_func(y_true, y_pred):
    scores = precision_recall_fscore_support(y_true, y_pred, average="macro")
    return scores[:3]

def loss_func(y_true, y_pred):
    return 1.0 - f1_score(y_true, y_pred)
    

def run_rfe1(lang_pair, results_fname, subset=None, step=100, folds=2):
    ambig_fname = config["sample"][lang_pair]["ambig_fname"]
    ambig_map = AmbiguityMap(ambig_fname, subset=subset)
    
    samples_fname = config["sample"][lang_pair]["samples_filt_fname"]
    sample_hdfile = h5py.File(samples_fname, "r")
    
    data_gen = DataSetGenerator(ambig_map, sample_hdfile)
    
    estimator = SGDClassifier()
    
    descriptor = [ ("lemma", "U32"),
                   ("pos", "U32"),
                   ("#cand", "i"),
                   ("#feats", "i"),
                   ("prec", "f"),
                   ("rec", "f"),
                   ("f-score", "f")] 
    results = np.zeros(len(ambig_map), dtype=descriptor)
    i = 0
    
    for data in data_gen:
        print i+1, data.source_lempos, 
        if not data.target_lempos:
            print "*** no samples ***"
            continue
        lemma, pos = data.source_lempos.rsplit("/", 1)
        n_cand = len(data.target_lempos)
        samples = data.samples.tocsr()
        # Fix scoring func
        rfecv = RFECV(estimator=estimator, 
                      step=step, 
                      cv=StratifiedKFold(data.targets, folds),
                      loss_func=loss_func,
                      verbose=True
                      )
        rfecv.fit(samples, data.targets)
        samples = rfecv.transform(samples)
        scores = cross_val_score(estimator, 
                                 samples, 
                                 data.targets,
                                 score_func=score_func)
        scores = scores.mean(axis=1) * 100
        results[i] = (lemma, pos, n_cand, rfecv.n_features_) + tuple(scores)
        print results[i]
        print rfecv.cv_scores_
        i += 1

    np.save(results_fname, results[:i])
    
    
    
if __name__ == "__main__":
    lang_pair = "de-en"
    results_fname = "rfe1_results_{}.npy".format(lang_pair) 
    run_rfe1(lang_pair, results_fname,
              subset = {"anmelden/v*.full", "Magazin/n"}
              )