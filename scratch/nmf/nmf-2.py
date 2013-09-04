#!/usr/bin/env python

"""
Attempt at Cross-Validated Results on Context Samples
using Non-negative Matrix Factorization.

However, 
- slow
- initial results are worse than without NMF

"""

import codecs
import sys

import numpy as np
import pylab as pl
import h5py

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import NMF
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFpr, chi2

from tg.config import config
from tg.ambig import AmbiguityMap
from tg.sample import DataSetGenerator
from tg.skl.selection import MinCountFilter, MaxFreqFilter



class Scorer:
    """
    Scoring function that tracks precision, recall and F scores for each
    fold. This a work-around for the problem that the cross_val_score
    function does not allow scoring functions that return more than one
    score.
    """
    
    def __init__(self, folds=3):
        self.scores = np.zeros((folds, 3))
        self.folds = folds
        self.fold_count = 0

    def __call__(self, estimator, X, y_true):
        y_pred = estimator.predict(X)
        scores = precision_recall_fscore_support(y_true, y_pred, average="macro")
        self.scores[self.fold_count] = scores[:3]
        self.fold_count += 1
        # return only F-score to cross_val_score
        return scores[2]
    
    def mean_scores(self):
        """
        return mean score over all folds, as percentage
        """
        assert self.fold_count == self.folds
        return self.scores.mean(axis=1) * 100
    
    
    

def run_cv1(lang_pair, results_fname, subset=None):
    ambig_fname = config["sample"][lang_pair]["ambig_fname"]
    ambig_map = AmbiguityMap(ambig_fname, subset=subset)
    
    samples_fname = config["sample"][lang_pair]["samples_filt_fname"]
    sample_hdfile = h5py.File(samples_fname, "r")
    
    data_gen = DataSetGenerator(ambig_map, sample_hdfile)
    
    descriptor = [ ("lemma", "U32"),
                   ("pos", "U32"),
                   ("#cand", "i"),
                   ("prec", "f"),
                   ("rec", "f"),
                   ("f-score", "f")] 
    results = np.zeros(len(ambig_map), dtype=descriptor)
    
    i = 0
    
    for data in data_gen:
        if not data.target_lempos:
            print "*", data.source_lempos, "no samples"
            continue
        print i+1, data.source_lempos
        lemma, pos = data.source_lempos.rsplit("/", 1)
        n_cand = len(data.target_lempos)
        # assume at least one topic per translation candidate
        n_components = n_cand + 1
        
        # Perform feature selaction, because otherwise NMF is way too slow
        classifier = Pipeline( [("MCF", MinCountFilter(5)),
                                ("MFF", MaxFreqFilter(0.05)), 
                                ("CHI2", SelectFpr(chi2, alpha=0.001 )),
                                ("NMF", NMF(n_components=n_components)),
                                ("MNB", MultinomialNB()),
                                #("SVM", LinearSVC()),
                                ])
        
        scorer = Scorer()
        cross_val_score(classifier, 
                        data.samples, 
                        data.targets,
                        scoring=scorer,
                        verbose=1)
        results[i] = (lemma, pos, n_cand) + tuple(scorer.mean_scores())
        i += 1
          
    np.save(results_fname, results[:i])

    
def write_table(results):
    results.sort(order="f-score")
    results = results[::-1]
    outf = codecs.getwriter('utf8')(sys.stdout)
    outf.write(u"{:32}{:32}{:>16}{:>16}{:>16}{:>16}\n".format(
        "LEMMA", "POS", "#CAND", "PREC", "REC", "F1"))
    line = (2 * 32 + 4 * 16) * "-" + "\n"
    outf.write(line)
    for r in results: 
        outf.write(u"{:32}{:32}{:16d}{:16.2f}{:16.2f}{:16.2f}\n".format(*r))
    outf.write(line)
    
    
def plot_1(results):
    """
    scatterplot of F1 score as a function of number of candidate translations
    """   
    pl.scatter(results["#cand"], results["f-score"])
    pl.xlim(0,50)
    pl.ylim(0,100)
    pl.xlabel('#Candidate translations')
    pl.ylabel('F1 score')
    pl.show()
    
def plot_2(results):
    """
    histogram of number of candidate translations
    """
    pl.hist(results["#cand"], bins=results["#cand"].max())
    pl.xlabel('#Candidate translations')
    pl.ylabel('#Source lemmas')
    pl.show()
    
def pos(results):
    print "{:16}{:>8}{:>8}{:>8}{:>8}".format("POS", "N", "PREC", "REC", "F1")
    line = (16 + 4*8) * "-"
    print line
    for pos in np.unique(results["pos"]):
        subres = results[results["pos"] == pos]
        print "{:16}{:8d}{:8.2f}{:8.2f}{:8.2f}".format(
            pos, 
            len(subres), 
            subres["prec"].mean(), 
            subres["rec"].mean(),\
            subres["f-score"].mean())
    print line
    
    
def plot_3(results): 
    n_cand = np.arange(2,51) 
    
    for pos in "n", "v*.full", "adj", "adv":
        scores = []    
        for n in n_cand:
            subres = results[results["#cand"] == n]
            subres = subres[subres["pos"] == pos]
            scores.append(subres["f-score"].mean())        
        pl.plot(n_cand, scores, "-o", label=pos)
        
    # baseline is F1 score for random classification with an n-class problem 
    baseline = 1.0 / n_cand * 100
    pl.plot(n_cand, baseline, "-o", label="baseline")
        
    pl.xlim(0,50)
    pl.ylim(0,100)
    pl.xlabel('#Candidate translations')
    pl.ylabel('F1 score')
    pl.legend()
    
    pl.show()
    
    
    
    
if __name__ == "__main__":
    lang_pair = "de-en"
    results_fname = "nmf-1_results_{}.npy".format(lang_pair) 
    run_cv1(lang_pair, results_fname,
            subset = {"anmelden/v*.full", "Magazin/n"}
            )
    results = np.load(results_fname)
    write_table(results)
    #plot_1(results)
    #plot_2(results)
    #pos(results)
    #plot_3(results)
    

