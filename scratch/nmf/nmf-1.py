#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Proof of concept using non-negtive matrix factorization 
for dimensionality reduction
"""


import h5py
import numpy as np

from sklearn.decomposition import NMF
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Binarizer
from tg.config import config
from tg.ambig import AmbiguityMap
from tg.sample import DataSetGenerator
from tg.skl.selection import MinCountFilter, MaxFreqFilter
from sklearn.feature_selection import SelectFpr, chi2

lang_pair = "de-en"
subset = {u"Strömung/n", "Magazin/n", "trauen/v*.full"}
n_samples = None #1000
n_top_words = 25

ambig_fname = config["sample"][lang_pair]["ambig_fname"]
ambig_map = AmbiguityMap(ambig_fname, subset=subset)

samples_fname = config["sample"][lang_pair]["samples_filt_fname"]
sample_hdfile = h5py.File(samples_fname, "r")

data_gen = DataSetGenerator(ambig_map, sample_hdfile)

preproc = Pipeline( [("MCF", MinCountFilter(5)),
                     ("MFF", MaxFreqFilter(0.05)), 
                     # TfIDF transformation fails because it cannot transform 
                     # the vocabulary, which is an array of strings
                     #("TFIDF", TfidfTransformer()),
                     ("CHI2", SelectFpr(chi2, alpha=0.001 )),
                     ])
                     


for data in data_gen:
    print data.source_lempos
    print data.target_lempos
    samples = data.samples.tocsr()[:n_samples]
    targets = data.targets[:n_samples]
    print samples.shape
    samples = preproc.fit_transform(samples, targets)
    vocab = preproc.transform(sample_hdfile["vocab"]).flatten()
    print samples.shape
    print vocab.shape
    # number of dimensions is one more than number of translations
    n_topics = len(data.target_lempos) + 1
    nmf = NMF(n_components=n_topics)
    nmf.fit(samples)
    
    for topic_idx, topic in enumerate(nmf.components_):
        print "Topic #%d:" % topic_idx
        print "\n".join(["{:.2f}    {}".format(topic[i], str(vocab[i]))
                        for i in topic.argsort()[:-n_top_words - 1:-1]])
        print
    
print    
    
    
   
    