#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
evaluate most-frequent-translation context vectors
with Nearest Centroid models
"""

import cPickle
import os
import logging
import sys

import numpy as np

from sklearn.feature_selection import SelectFpr,SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

from tg.config import config
from tg.utils import set_default_log, makedirs, text_table
from tg.classify import TranslationClassifier
from tg.ambig import AmbiguityMap
from tg.model import ModelBuilder
from tg.classcore import ClassifierScore, filter_functions, Vectorizer
from tg.exps.postproc import postprocess
from tg.bestscore import BestScorer
from tg.skl.selection import MinCountFilter, MaxFreqFilter

log = logging.getLogger(__name__)



def nb_exp(data_sets=config["eval"]["data_sets"], 
                lang_pairs=(),
                text=False,
                draw=False,
                diff=False,
                trash_models=False):
    
    descriptor = [ ("data", "S16"),
                   ("lang", "S8"),
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
            
            name = "{}_{}_{}".format(
                script_fname, data, lang)
            exp_dir = "_" + name     
            if not os.path.exists(exp_dir):
                os.makedirs(exp_dir)
            models_fname = exp_dir + "/" + name + ".hdf5"
            ##classifier = Pipeline( [("MCF", MinCountFilter(5)),
                                    ##("MFF", MaxFreqFilter(0.01)),
                                    ###("CHI2", SelectKBest(chi2)),
                                    ##("CHI2", SelectFpr(chi2)),
                                    ###("TFIDF", TfidfTransformer()),
                                    ##("MNB", MultinomialNB())
                                    ##])
            classifier = MultinomialNB()
            
            # get ambiguity map
            ambig_map = AmbiguityMap(ambig_fname, graphs_fname=graphs_fname)

            # train classifier
            model_builder = ModelBuilder( ambig_map, samples_fname,
                                          models_fname, classifier, 
                                          #with_vocab_mask=True
                                          )
            model_builder.run()
            
            # apply classifier
            model = TranslationClassifier(models_fname)
            score_attr="nb_score"
            source_lang = lang.split("-")[0]
            # use max MFT score for building translation vectors 
            vectorizer = Vectorizer(score_attr="freq_score")
            scorer = ClassifierScore(model,
                                     filter=filter_functions(source_lang),
                                     vectorizer=vectorizer)
            graph_list = cPickle.load(open(graphs_fname))
            scorer(graph_list)
            
            best_scorer = BestScorer(["nb_score", "freq_score"])
            best_scorer(graph_list)
            
            scored_graphs_fname = exp_dir + "/" + name + "_graphs.pkl"
            log.info("saving scored graphs to " + scored_graphs_fname)
            cPickle.dump(graph_list, open(scored_graphs_fname, "w"))
            #graph_list = cPickle.load(open(scored_graphs_fname))
            
            nist_score, bleu_score = postprocess(
                name, data, lang, graph_list, 
                best_score_attr="best_score",
                base_score_attrs=["nb_score","freq_score"],
                out_dir=exp_dir,
                base_fname=name,
                text=text,
                draw=draw,
                diff=diff
            ) 
            
            results[exp_count] = (data, lang, nist_score, bleu_score, name)
            results_fname = exp_dir + "/" + name + ".npy"
            log.info("saving result to " + results_fname)
            np.save(results_fname, results[exp_count])
            exp_count += 1
            
            if trash_models:
                log.info("Trashing models file " + models_fname)
                os.remove(models_fname)
            
            # add to table of results per data set & language pair
            sub_results = results[(results["lang"] == lang) &
                                  (results["data"] == data)]
            sub_results = np.sort(sub_results, 
                                  axis=0, 
                                  order=("lang", "blue"))[::-1]
            text_table(sub_results, results_outf)
            results_outf.write("\n\n")
            
    results_outf.close()
    results = results[:exp_count]       
    results_fname = "_" + script_fname + "_results.npy"    
    log.info("saving pickled results to " + results_fname)
    np.save(results_fname, results)
    text_table(results)
    
    return results
        
        
        
# for logging to stderr in utf-8 use:
set_default_log(level=logging.INFO)

#logging.getLogger("tg.classcore").setLevel(logging.DEBUG)    


nb_exp(
    data_sets=("metis", "presemt-dev"),
    # lang_pairs = ("de-en",),
    #lang_pairs=("no-en", "no-de"),
    #lang_pairs=("de-en", "en-de"),
    text=True,
    #draw=True,
    diff=True,
    #trash_models=True,
)

