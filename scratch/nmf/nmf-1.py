#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
#subset = {"anmelden/v*.full", "Magazin/n"}
subset = {"Magazin/n"}
n_topics = 10
n_samples = None #1000
n_top_words = 25

ambig_fname = config["sample"][lang_pair]["ambig_fname"]
ambig_map = AmbiguityMap(ambig_fname, subset=subset)

samples_fname = config["sample"][lang_pair]["samples_filt_fname"]
sample_hdfile = h5py.File(samples_fname, "r")
vocab = sample_hdfile["vocab"]

data_gen = DataSetGenerator(ambig_map, sample_hdfile)

preproc = Pipeline( [#("MCF", MinCountFilter(50)),
                     ("MFF", MaxFreqFilter(0.05)), 
                     #("TFIDF", TfidfTransformer()),
                     ("CHI2", SelectFpr(chi2, alpha=0.001 )),
                     ])
                     
nmf = NMF(n_components=n_topics)


for data in data_gen:
    print data.source_lempos
    print data.target_lempos
    samples = data.samples.tocsr()[:n_samples]
    targets = data.targets[:n_samples]
    print samples.shape
    samples = preproc.fit_transform(samples, targets)
    vocab = preproc.transform(vocab).flatten()
    print samples.shape
    print vocab.shape
    nmf.fit(samples)
    
    
    for topic_idx, topic in enumerate(nmf.components_):
        print "Topic #%d:" % topic_idx
        print "\n".join(["{:.2f}    {}".format(topic[i], str(vocab[i]))
                        for i in topic.argsort()[:-n_top_words - 1:-1]])
        print
    
print    
    
    
   
    