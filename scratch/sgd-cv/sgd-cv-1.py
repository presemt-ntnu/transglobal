#!/usr/bin/env python

"""
cross-validated results on context samples with SGD
"""

import codecs
import sys
import logging

import numpy as np
import pylab as pl
import asciitable as at
import h5py

from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tg.exps.support import grid_search

from tg.config import config
from tg.ambig import AmbiguityMap
from tg.sample import DataSetGenerator
from tg.utils import set_default_log

log = logging.getLogger(__name__)


class Scorer:
    """
    Scoring function that tracks precision, recall, F1 and accuracy scores
    for each fold. This a work-around for the problem that the
    cross_val_score function does not allow scoring functions that return
    more than one score.
    """
    
    def __init__(self, folds=3, average="macro"):
        self.folds = folds
        self.average = average
        n_metrics = 4
        self.scores = np.zeros((folds, n_metrics))
        self.fold_count = 0

    def __call__(self, estimator, X, y_true):
        y_pred = estimator.predict(X)
        prec, rec, f, _ = precision_recall_fscore_support(y_true, y_pred, 
                                                          average=self.average)
        acc = accuracy_score(y_true, y_pred)
        self.scores[self.fold_count] = prec, rec, f, acc
        self.fold_count += 1
        # return only F-score to cross_val_score
        return f
    
    def mean_scores(self):
        """
        return mean score over all folds, as percentage
        """
        assert self.fold_count == self.folds
        return self.scores.mean(axis=0) * 100
    
    
   

@grid_search
def sgd_classifier(alpha=0.0001, class_weight=None, epsilon=0.1, eta0=0.0,
                   fit_intercept=True, l1_ratio=0.15, learning_rate='optimal', 
                   loss='hinge', n_iter=5, n_jobs=1, penalty='l2', power_t=0.5, 
                   random_state=None, rho=None, shuffle=False, verbose=0, 
                   warm_start=False):
    return SGDClassifier(alpha=alpha, class_weight=class_weight, 
                         epsilon=epsilon, eta0=eta0, 
                         fit_intercept=fit_intercept, l1_ratio=l1_ratio, 
                         learning_rate=learning_rate, loss=loss, n_iter=n_iter, 
                         n_jobs=n_jobs, penalty=penalty, power_t=power_t,
                         random_state=random_state, rho=rho, shuffle=shuffle, 
                         verbose=verbose, warm_start=warm_start)
    
     

def run_cv1(lang_pair, results_fname, subset=None):
    ambig_fname = config["sample"][lang_pair]["ambig_fname"]
    ambig_map = AmbiguityMap(ambig_fname, subset=subset)
    
    samples_fname = config["sample"][lang_pair]["samples_filt_fname"]
    sample_hdfile = h5py.File(samples_fname, "r")
    
    data_gen = DataSetGenerator(ambig_map, sample_hdfile)
    
    classifiers = list(sgd_classifier(
        _alpha = (0.00001, 0.0001, 0.001),
        _loss = ("hinge", "log"),
        _n_iter = (5, 10),
        _penalty = ("l1", "l2"),
        shuffle = True,              # shuffle seems always benificial
        random_state = 73761232569,  # but needs to be repeatable
        n_jobs = 10,
        
    ))
    
    descriptor = [ ("lemma", "S32"),
                   ("pos", "S32"),
                   ("#cand", "i"),
                   ("alpha", "f"),
                   ("loss", "S16"),
                   ("n_iter", "i"),
                   ("penalty", "S16"),
                   ("prec", "f"),
                   ("rec", "f"),
                   ("f-score", "f"),
                   ("accuracy", "f")] 
    results = np.zeros(9999, dtype=descriptor)

    i = 0
    
    for n, data in enumerate(data_gen):
        if not data.target_lempos:
            log.error(data.source_lempos + u"no samples")
            continue
        log.info(u"{}/{} {}".format(n+1, len(ambig_map), data.source_lempos))
        lemma, pos = data.source_lempos.rsplit("/", 1)
        n_cand = len(data.target_lempos)
        # *** shuffling is essential for SGD! *** 
        samples, targets = shuffle(data.samples, data.targets)
        
        for classifier in classifiers:
            scorer = Scorer()
            cross_val_score(classifier, 
                            samples, 
                            targets,
                            scoring=scorer)  
            params = (repr(lemma), # HACK: 
                      pos,         # asciitable can't handle unicode 
                      n_cand,
                      classifier.alpha,
                      classifier.loss,
                      classifier.n_iter,
                      classifier.penalty)
            results[i] =  params + tuple(scorer.mean_scores())
            i += 1
            np.save(results_fname, results[:i])
            at.write(results[:i], 
                     results_fname.replace(".npy", ".txt"),
                     Writer=at.FixedWidthTwoLine, 
                     delimiter_pad=" ")
            


if __name__ == "__main__":
    lang_pair = "de-en"
    set_default_log(log_fname="_sgd-cv-1_results_{}.log".format(lang_pair))
    results_fname = "_sgd-cv-1_results_{}.npy".format(lang_pair) 
    run_cv1(lang_pair, results_fname,
            #subset = {"anmelden/v*.full", "Magazin/n"}
            )
