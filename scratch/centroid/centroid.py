#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
evaluate Nearest Centroid models
"""

import cPickle
import os
import logging
import sys

import numpy as np
import asciitable as at

from sklearn.feature_selection import SelectFpr, chi2, SelectKBest
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

from tg.config import config
from tg.utils import set_default_log, makedirs
from tg.classify import TranslationClassifier
from tg.model import ModelBuilder
from tg.classcore import ClassifierScore, filter_functions
from tg.exps.postproc import postprocess
from tg.bestscore import BestScore
from tg.skl.centroid import CosNearestCentroid, print_centroids, NearestCentroidProb
from tg.skl.selection import MinCountFilter, MaxFreqFilter

log = logging.getLogger(__name__)



def centroid_exp(data_sets=config["eval"]["data_sets"], 
                lang_pairs=(),
                text=False,
                draw=False,
                diff=False,
                trash_models=False,
                dump_centroids=False):
    
    descriptor = [ ("data", "S16"),
                   ("lang", "S8"),
                   ("min_count", "f"),
                   ("max_freq", "f"),
                   ("nist", "f"),
                   ("blue", "f"),
                   ("name", "S256") ] 
    results = np.zeros(9999, dtype=descriptor)
    exp_count = 0    
    script_fname = os.path.splitext(os.path.basename(__file__))[0]
    results_fname = "_" + script_fname + "_results.txt"
    results_outf = open(results_fname, "w")    
    
    for data in data_sets: 
        for lang in  lang_pairs or config["eval"][data].keys():
            ambig_fname = config["sample"][lang]["ambig_fname"]
            try:
                samples_fname = config["sample"][lang]["samples_filt_fname"]
            except KeyError:
                samples_fname = config["sample"][lang]["samples_fname"]
                log.warn("backing off to unfiltered samples from " + 
                         samples_fname)
            graphs_fname = config["eval"][data][lang]["graphs_fname"]
            
            #for min_count in (1, 5, 10, 25, 50, 100, 250, 1000, 2500, 5000):
            #    for max_freq in (0.0001, 0.001, 0.005, 0.01, 0.05, 0.10, 0.25, 0.5, 1.0):
            for min_count in (5,):
                for max_freq in (0.01,):
                    name = "{}_{}_{}_min_count={:d}_max_freq={:f}".format(
                        script_fname, data, lang, min_count, max_freq)
                    exp_dir = "_" + name     
                    if not os.path.exists(exp_dir):
                        os.makedirs(exp_dir)
                    models_fname = exp_dir + "/" + name + ".hdf5"
                    classifier = Pipeline( [("MCF", MinCountFilter(min_count)),
                                            ("MFF", MaxFreqFilter(max_freq)),
                                            ("CHI2", SelectFpr()),
                                            #("TFIDF", TfidfTransformer()),
                                            ("CNC", CosNearestCentroid())
                                            #("NC", NearestCentroidProb())
                                            ])
        
                    # train classifier
                    model_builder = ModelBuilder( 
                        ambig_fname, samples_fname, models_fname, classifier,
                        graphs_fname, with_vocab_mask=True)
                    model_builder.run()
                    
                    # print the centroids to a file, only the 50 best features
                    if dump_centroids:
                        print_fname = exp_dir + "/" + name + "_centroids.txt"
                        print_centroids(models_fname, 
                                        n=50, 
                                        outf=print_fname)
                
                    # apply classifier
                    model = TranslationClassifier(models_fname)
                    score_attr="centroid_score"
                    source_lang = lang.split("-")[0]
                    scorer = ClassifierScore(model,
                                             score_attr=score_attr,
                                             filter=filter_functions(source_lang))
                    graph_list = cPickle.load(open(graphs_fname))
                    scorer(graph_list)
                    
                    best_scorer = BestScore(["centroid_score", "freq_score"])
                    best_scorer(graph_list)
                    
                    scored_graphs_fname = exp_dir + "/" + name + "_graphs.pkl"
                    log.info("saving scored graphs to " + scored_graphs_fname)
                    cPickle.dump(graph_list, open(scored_graphs_fname, "w"))
                    #graph_list = cPickle.load(open(scored_graphs_fname))
                    
                    nist_score, bleu_score = postprocess(
                        name, data, lang, graph_list, 
                        best_score_attr="best_score",
                        base_score_attrs=["centroid_score","freq_score"],
                        out_dir=exp_dir,
                        base_fname=name,
                        text=text,
                        draw=draw,
                        diff=diff
                    ) 
                    
                    results[exp_count] = (data, lang, min_count, max_freq,
                                          nist_score, bleu_score, name)
                    results_fname = exp_dir + "/" + name + ".npy"
                    log.info("saving result to " + results_fname)
                    np.save(results_fname, results[exp_count])
                    exp_count += 1
                    
                    if trash_models:
                        log.info("Trashing models file " + models_fname)
                        os.remove(models_fname)
            
            
            sub_results = results[(results["lang"] == lang) &
                                  (results["data"] == data)]
            sub_results = np.sort(sub_results, 
                                  axis=0, 
                                  order=("lang", "blue"))[::-1]
            at.write(sub_results, results_outf, Writer=at.FixedWidthTwoLine, 
                     delimiter_pad=" ")
            results_outf.write("\n\n")
            
    results_outf.close()
    results = results[:exp_count]       
    results_fname = "_" + script_fname + "_results.npy"    
    log.info("saving pickled results to " + results_fname)
    np.save(results_fname, results)
    
    at.write(results, sys.stdout, Writer=at.FixedWidthTwoLine,
             delimiter_pad=" ")   
    
    return results
        
        
        
# for logging to stderr in utf-8 use:
set_default_log(level=logging.INFO)

#logging.getLogger("tg.skl.selection").setLevel(logging.DEBUG)    


centroid_exp(data_sets=("metis", "presemt-dev"),
             #lang_pairs=("no-en", "no-de"),
             #lang_pairs=("de-en", "en-de"),
             #text=True,
             draw=True,
             diff=True,
             trash_models=True,
             dump_centroids=True,
             )

