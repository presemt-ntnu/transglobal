#!/usr/bin/env python

"""
CV1: Cross-Validated Results on Context Samples

See iPython notebook named cv1.ipnb
"""

import codecs
import sys

import numpy as np
import pylab as pl
import h5py

from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from tg.config import config
from tg.ambig import AmbiguityMap
from tg.sample import DataSetGenerator


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
    
    
    

def run_cv1(lang_pair, results_fname, subset=None):
    ambig_fname = config["sample"][lang_pair]["ambig_fname"]
    ambig_map = AmbiguityMap(ambig_fname, subset=subset)
    
    samples_fname = config["sample"][lang_pair]["samples_filt_fname"]
    sample_hdfile = h5py.File(samples_fname, "r")
    
    data_gen = DataSetGenerator(ambig_map, sample_hdfile)
    
    classifier = MultinomialNB()
    
    descriptor = [ ("lemma", "U32"),
                   ("pos", "U32"),
                   ("#cand", "i"),
                   ("prec", "f"),
                   ("rec", "f"),
                   ("f-score", "f"),
                   ("accuracy", "f")] 
    results = np.zeros(len(ambig_map), dtype=descriptor)
    
    i = 0
    
    for data in data_gen:
        if not data.target_lempos:
            print "*", data.source_lempos, "no samples"
            continue
        print i+1, data.source_lempos
        lemma, pos = data.source_lempos.rsplit("/", 1)
        n_cand = len(data.target_lempos)
        scorer = Scorer()
        cross_val_score(classifier, 
                        data.samples, 
                        data.targets,
                        scoring=scorer)        
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
    
def plot_2(results, fname=None):
    if fname:
        pl.figure(figsize=(8,4), dpi=1200)        
    else:
        pl.figure(figsize=(8,4))
    pl.subplot(1,2,1)
    plot_cand_hist(results)
    pl.subplot(1,2,2)
    plot_mean_f1(results)
    pl.tight_layout()
    if fname:
        pl.savefig(fname, dpi=1200)
    else:
        pl.show()
    
def plot_mean_f1(results):
    """
    line plot of F1 score as a function of number of candidate translations
    """  
    n_cand = np.arange(2,51)
    f_scores = np.zeros(49)
    errors = np.zeros(49)
    for n in n_cand:
        subset = results[results["#cand"] == n]
        f_scores[n-2] = subset["f-score"].mean()
        errors[n-2] = subset["f-score"].std()
    pl.errorbar(n_cand, f_scores, fmt="-", yerr=errors/2.0, elinewidth=1,
                markersize=0, linewidth=2)
    pl.xlim(0,50)
    pl.ylim(0,100)
    pl.xlabel('#Candidate translations')
    pl.ylabel('F-score (%)')
    pl.grid()
    
def plot_cand_hist(results):
    """
    histogram of number of candidate translations
    """
    pl.hist(results["#cand"], bins=results["#cand"].max())
    pl.xlabel('#Candidate translations')
    pl.ylabel('#Source lemmas')
    pl.grid(axis="y")
    
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
    
def pos_latex(results):
    print "{:16} & {:>8} & {:>8} & {:>8} & {:>8} & {:>8} \\\\".format(
        "POS", "\\#Instances", "Precision", "Recall", "$F_1$", "Accuracy")
    line = (16 + 4*8) * "-"
    print line
    for pos in np.unique(results["pos"]):
        subres = results[results["pos"] == pos]
        print "{:16} & {:8d} & {:8.2f} & {:8.2f} & {:8.2f} & {:8.2f} \\\\".format(
            pos, 
            len(subres), 
            subres["prec"].mean(), 
            subres["rec"].mean(),\
            subres["f-score"].mean(),
            subres["accuracy"].mean())
    print line
    
    
def plot_3(results, fname=None): 
    if fname:
        pl.figure(figsize=(8,4), dpi=1200)        
    else:
        pl.figure(figsize=(6,4))
    #cmap = pl.cm.rainbow
    #n_colors = 5
    #colors = [cmap(i) for i in np.linspace(0.0, 0.8, n_colors)]
    #pl.rcParams["axes.color_cycle"] = colors
    pl.rcParams["legend.fontsize"] = 'medium'                     
    
    n_cand = np.arange(2,51) 
    
    for pos, label in [("n", "Nouns"),  ("v*.full", "Verbs"), 
                       ("adj", "Adjectives"), ("adv", "Adverbs")]:
        scores = []    
        for n in n_cand:
            subres = results[results["#cand"] == n]
            subres = subres[subres["pos"] == pos]
            scores.append(subres["f-score"].mean())        
        pl.plot(n_cand, scores, "-o", markersize=5, linewidth=2, label=label,
                markeredgewidth=0)
        
    # baseline is F1 score for random classification with an n-class problem 
    baseline = 1.0 / n_cand * 100
    pl.plot(n_cand, baseline, ":o", markersize=5, linewidth=2,
            label="Baseline", markeredgewidth=0)
        
    pl.xlim(0,50)
    pl.ylim(0,100)
    pl.xlabel('#Candidate translations')
    pl.ylabel('F-score (%)')
    pl.grid()
    pl.legend()
    
    pl.tight_layout()
    if fname:
        pl.savefig(fname, dpi=1200)
    else:
        pl.show()
    
    
    
    
if __name__ == "__main__":
    lang_pair = "de-en"
    results_fname = "cv1_results_{}.npy".format(lang_pair) 
    #run_cv1(lang_pair, results_fname,
    #        ##subset = {"anmelden/v*.full", "Magazin/n"}
    #        )
    results = np.load(results_fname)
    #plot_2(results, "_plot_2.eps")
    #write_table(results)
    #plot_1(results)
    #plot_2(results)
    #pos_latex(results)
    plot_3(results, "_plot3.eps")
    

