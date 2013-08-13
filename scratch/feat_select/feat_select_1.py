#!/usr/bin/env python

"""
feature selection
"""

import codecs
import sys
import random

import numpy as np
import pylab as pl
#import pylab as pl
import h5py

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_selection.univariate_selection import SelectFpr, chi2
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import f1_score, precision_recall_fscore_support

from tg.config import config
from tg.ambig import AmbiguityMap
from tg.sample import DataSetGenerator


def score_func(y_true, y_pred):
    scores = precision_recall_fscore_support(y_true, y_pred, average="macro")
    return scores[:3]


class MySelectFpr(SelectFpr):
    
    total_n_feats = None
    
    def fit(self, X, y):
        SelectFpr.fit(self, X,y)
        MySelectFpr.total_n_feats += self.get_support().sum()
        return self
    
    @classmethod
    def reset_total_n_feats(cls):
        cls.total_n_feats = 0.0
        
    @classmethod
    def get_total_n_feats(cls):
        return cls.total_n_feats
        
    
    
        
    

def ambig_map_sample(lang_pair, k=0, r=1.0, pos=set()):
    """
    take sample from orignal ambiguity map,
    either an absulute number or a fraction,
    possibly restricted to particular POS tags
    """
    ambig_fname = config["sample"][lang_pair]["ambig_fname"]
    ambig_map = AmbiguityMap(ambig_fname)
    
    if pos:
        select = [ sl for sl in ambig_map.source_iter()
                   if sl.rsplit("/",1)[1] in pos ]    
    else:
        select =  list(ambig_map.source_iter())
        
    if r < 1.0:
        k = int(round(len(select) * r))
    elif k == 0:
        k = len(select)
    
    select = random.sample(select, k)
    
    ambig_map.source_target_map = dict( (sl, ambig_map[sl])
                                        for sl in select )
    
    return ambig_map


def feat_select_1(lang_pair, ambig_map, results_fname):
    samples_fname = config["sample"][lang_pair]["samples_filt_fname"]
    sample_hdfile = h5py.File(samples_fname, "r")
    
    data_gen = DataSetGenerator(ambig_map, sample_hdfile)
    
    descriptor = [ ("lemma", "U32"),
                   ("pos", "U32"),
                   ("#cand", "i"),
                   ("#feats", "i"),
                   ("prec", "f"),
                   ("rec", "f"),
                   ("f-score", "f")] 
    results = np.zeros(len(ambig_map), dtype=descriptor)
    
    i = 0
    folds = 3
    
    classifier = Pipeline( [("CHI2", MySelectFpr(chi2)),
                            ("MNB", MultinomialNB())
                            ])
    
    for data in data_gen:
        if not data.target_lempos:
            print "*", data.source_lempos, "no samples"
            continue
        print i+1, data.source_lempos
        lemma, pos = data.source_lempos.rsplit("/", 1)
        n_cand = len(data.target_lempos)
        MySelectFpr.reset_total_n_feats()
        scores = cross_val_score(classifier, 
                                 data.samples, 
                                 data.targets,
                                 score_func=score_func)
        scores = scores.mean(axis=1) * 100
        n_feats = MySelectFpr.get_total_n_feats() / folds
        results[i] = (lemma, pos, n_cand, n_feats) + tuple(scores)
        i += 1
          
    np.save(results_fname, results[:i])
    

def write_table(results):
    results.sort(order="f-score")
    results = results[::-1]
    outf = codecs.getwriter('utf8')(sys.stdout)
    outf.write(u"{:32}{:8}{:>8}{:>12}{:>8}{:>8}{:>8}\n".format(
        "LEMMA", "POS", "#CAND", "#FEATS", "PREC", "REC", "F1"))
    line = (32 + 12 + 5 * 8) * "-" + "\n"
    outf.write(line)
    for r in results: 
        outf.write(u"{:32}{:8}{:8d}{:12d}{:8.2f}{:8.2f}{:8.2f}\n".format(*r))
    outf.write(line)    
    outf.write(u"{:32}{:8}{:>8.2f}{:>12.0f}{:>8.2f}{:>8.2f}{:>8.2f}\n".format(
        "", 
        "", 
        results["#cand"].mean(),
        results["#feats"].mean(), 
        results["prec"].mean(), 
        results["rec"].mean(), 
        results["f-score"].mean()))
    

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
    results_fname = "feat_select_1_results_{}.npy".format(lang_pair) 
       
    ambig_map = ambig_map_sample(lang_pair,
                                 #k = 5,
                                 #r=0.1, 
                                 #pos={"n", "v*.full", "adj", "adv"})
                                 #pos={"n"}
                                 )
    print ambig_map.source_target_map.keys()
                                 
    feat_select_1(lang_pair, ambig_map, results_fname
                  )
    results = np.load(results_fname)
    write_table(results)