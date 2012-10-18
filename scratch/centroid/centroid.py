#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
evaluate Nearest Centroid models
"""

import cPickle
import os
import logging

import numpy as np

from sklearn.feature_selection import SelectFpr, chi2
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

from tg.config import config
from tg.utils import set_default_log, makedirs
from tg.classify import TranslationClassifier
from tg.model import ModelBuilder
from tg.classcore import ClassifierScore, filter_functions
from tg.exps.postproc import postprocess
from tg.bestscore import BestScore
from tg.skl.centroid import CosNearestCentroid, print_centroids
from tg.skl.selection import MinCountFilter, MaxFreqFilter

log = logging.getLogger(__name__)



def centroid_exp(data_sets=config["eval"]["data_sets"], 
                lang_pairs=()):
    
    descriptor = [ ("data", "S16"),
                   ("lang", "S8"),
                   ("nist", "f"),
                   ("blue", "f"),
                   ("name", "S128") ] 
    results = np.zeros(9999, dtype=descriptor)
    exp_count = 0
    
    for data in data_sets: 
        for lang in  lang_pairs or config["eval"][data].keys():
            ambig_fname = config["sample"][lang]["ambig_fname"]
            samples_fname = config["sample"][lang]["samples_filt_fname"]
            #samples_fname = config["sample"][lang]["samples_fname"]
            graphs_fname = config["eval"][data][lang]["graphs_fname"]
            script_fname = os.path.splitext(os.path.basename(__file__))[0]
            name = "{}_{}_{}".format(script_fname, data, lang)
            exp_dir = "_" + name     
            if not os.path.exists(exp_dir):
                os.makedirs(exp_dir)
            models_fname = exp_dir + "/" + name + ".hdf5"
            classifier = CosNearestCentroid()
            classifier = Pipeline( [("MCF", MinCountFilter()),
                                    ("MFF", MaxFreqFilter()),
                                    #("CHI2", SelectFpr(chi2)),
                                    #("TFIDF", TfidfTransformer()),
                                    ("CNC", CosNearestCentroid())
                                    #("NC", NearestCentroidProb())
                                    ])

            # train classifier
            model_builder = ModelBuilder( 
                ambig_fname, samples_fname, models_fname, classifier,
                graphs_fname, with_vocab_mask=True)
            model_builder.run()
            
            # print the centroids to a file, only for nouns, only the 50 best
            # features
            print_fname = exp_dir + "/" + name + "_centroids.txt"
            print_centroids(models_fname, 
                            #pos="n", 
                            n=50, 
                            outf=print_fname)
        
            # apply classifier
            model = TranslationClassifier(models_fname)
            score_attr="centroid_score"
            source_lang = lang.split("-")[0]
            scorer = ClassifierScore(model,
                                     score_attr=score_attr,
                                     # FIXME: filter function for Norwegian!
                                     filter=filter_functions(source_lang)
                                     )
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
                draw=True
            ) 
            
            results[exp_count] = (data, lang, nist_score, bleu_score, name)
            exp_count += 1
            
    results = results[:exp_count]    
    results_fname = exp_dir + "/" + name + ".npy"
    log.info("saving results to " + results_fname)
    np.save(results_fname, results)
    
    print "%-16s\t%-8s\t%8s\t%8s\t%s" % ("DATA:", "LANG:", 
                                         "NIST:", "BLUE:", "NAME:")
    for row in results:
        print "%-16s\t%-8s\t%8.4f\t%8.4f\t%s" % tuple(row)
        
    return results
        


# for logging to stderr in utf-8 use:
set_default_log(level=logging.INFO)

# logging.getLogger("model").setLevel(logging.DEBUG)    

#centroid_exp()
centroid_exp(data_sets=("metis", "presemt-dev"),
             lang_pairs=("de-en", "en-de")
             )

