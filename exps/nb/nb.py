#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
evaluate Nearest Centroid models
"""

import collections
import cPickle
import os
import logging
import sys


import numpy as np
import asciitable as at

from sklearn.feature_selection import SelectFpr,SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

from tg.config import config
from tg.utils import set_default_log, makedirs
from tg.classify import TranslationClassifier
from tg.ambig import AmbiguityMap
from tg.model import ModelBuilder, PriorModelBuilder
from tg.classcore import ClassifierScore, filter_functions
from tg.exps.postproc import postprocess
from tg.bestscore import BestScore
from tg.skl.selection import MinCountFilter, MaxFreqFilter

log = logging.getLogger(__name__)



def run(name,
              classifier,
              data_sets=config["eval"]["data_sets"], 
              lang_pairs=(),
              score_attr=None,
              save_graphs=False,
              save_models=False,
              text=False,
              diff=False,
              draw=False,
              n_graphs=None):
    
    descriptor = [ ("data", "S16"),
                   ("lang", "S8"),
                   ("nist", "f"),
                   ("blue", "f"),
                   ("name", "S256") ] 
    results = np.zeros(9999, dtype=descriptor)
    results_fname = "_" + name + "_results.npy"
    
    exps = multi_exp(name,
                     classifier,
                     data_sets=data_sets,
                     lang_pairs=lang_pairs,
                     score_attr=score_attr,
                     save_graphs=save_graphs,
                     save_models=save_models,
                     text=text,
                     diff=diff,
                     draw=draw,
                     n_graphs=n_graphs,
                     builder_class=PriorModelBuilder)
    
    for exp_count, result in enumerate(exps): 
        results[exp_count] = result 
        log.info("saving pickled results to " + results_fname)
        # immediately save each new result
        np.save(results_fname, results[:exp_count + 1])  

    return results[:exp_count + 1]   


def single_exp(name, data, lang, classifier, score_attr=None,
               save_graphs=False, save_models=False, text=False, 
               diff=False, draw=False, n_graphs=None,
               builder_class=ModelBuilder):
    exp_name = "{}_{}_{}".format(name, data, lang)
    exp_dir = "_" + exp_name
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)  
    fname_prefix = os.path.join(exp_dir, exp_name)
    
    source_lang, target_lang = lang.split("-")    
        
    # get graphs
    graphs_fname = config["eval"][data][lang]["graphs_fname"]
    graphs = cPickle.load(open(graphs_fname))[:n_graphs]                
    
    # get ambiguity map
    ambig_fname = config["sample"][lang]["ambig_fname"]            
    ambig_map = AmbiguityMap(ambig_fname, graphs=graphs)

    # train disambiguation models            
    try:
        samples_fname = config["sample"][lang]["samples_filt_fname"]
    except KeyError:
        samples_fname = config["sample"][lang]["samples_fname"]
        log.warn("backing off to unfiltered samples from " + 
                 samples_fname)            
    models_fname = fname_prefix + "_models.hdf5"
    # counts filename is only required in combination with PriorModelBuilder
    # and is ignored otherwise
    counts_fname = config["count"]["lemma"][target_lang]["pkl_fname"]         
    model_builder = builder_class( ambig_map, samples_fname,
                                   models_fname, classifier, 
                                   with_vocab_mask=True,
                                   counts_fname=counts_fname)
    model_builder.run()
    
    # apply classifier
    model = TranslationClassifier(models_fname)
    score_attr = score_attr or name + "_score"
    scorer = ClassifierScore(model,
                             score_attr=score_attr,
                             filter=filter_functions(source_lang))
    scorer(graphs)
    if not save_models:
        log.info("Trashing models file " + models_fname)
        os.remove(models_fname)            
    
    # determine overall score
    best_scorer = BestScore([score_attr, "freq_score"])
    best_scorer(graphs)
    
    # save annotated graphs
    if save_graphs:
        scored_graphs_fname = fname_prefix + "_graphs.pkl"
        log.info("saving scored graphs to " + scored_graphs_fname)
        cPickle.dump(graphs, open(scored_graphs_fname, "w"))
    
    # post-process
    nist_score, bleu_score = postprocess(
        name, data, lang, graphs, 
        best_score_attr="best_score",
        base_score_attrs=[score_attr, "freq_score"],
        out_dir=exp_dir,
        base_fname=exp_name,
        text=text,
        draw=draw,
        diff=diff)
    
    log.info("NIST={:.3f}, BLEU={:.3f}".format(nist_score, bleu_score))
    return bleu_score, nist_score
        


def multi_exp(name,
              classifier,
              data_sets=config["eval"]["data_sets"], 
              lang_pairs=(),
              score_attr=None,
              save_graphs=False,
              save_models=False,
              text=False,
              diff=False,
              draw=False,
              n_graphs=None,
              single_exp=single_exp,
              builder_class=ModelBuilder):
    
    for data in data_sets: 
        for lang in lang_pairs or config["eval"][data].keys():
                
            bleu_score, nist_score = single_exp(name, data, lang, classifier, 
                                                score_attr=score_attr,
                                                save_graphs=save_graphs,
                                                text=text,
                                                diff=diff,
                                                draw=draw,
                                                n_graphs=n_graphs,
                                                builder_class=builder_class)
            
            yield data, lang, bleu_score, nist_score, name




        
        
# for logging to stderr in utf-8 use:
set_default_log(level=logging.INFO)

#logging.getLogger("tg.classcore").setLevel(logging.DEBUG)    

classifier = Pipeline( [("MCF", MinCountFilter(5)),
                        ("MFF", MaxFreqFilter(0.01)),
                        #("CHI2", SelectKBest(chi2)),
                        ("CHI2", SelectFpr(chi2)),
                        #("TFIDF", TfidfTransformer()),
                        ("MNB", MultinomialNB())
                        ])

run(
    name="nb",
    data_sets=("metis",
               #"presemt-dev",
               ),
    lang_pairs=("de-en",
                "en-de",
                ),
    classifier=classifier,
    #score_attr="nb",
    #save_graphs=True, 
    #save_models=True, 
    #text=True, 
    #diff=True,
    #draw=True, 
    n_graphs=2
)

