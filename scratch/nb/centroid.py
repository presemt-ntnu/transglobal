#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
evaluate Nearest Centroid models on Presemt evaluation data
"""

import logging
import os
from os.path import join

import numpy as np

from sklearn.neighbors import NearestCentroid
from sklearn.feature_selection import SelectFpr, chi2

from tg.config import config
from tg.utils import set_default_log
from tg.classify import NearestCentroidClassifier
from tg.model import NearestCentroidModelBuilder


# THIS IS JUST A QUICK HACK
# score_model should be shared among NB and Centroid

import cPickle
from tg.classcore import ClassifierScore, filter_functions
from tg.arrange import Arrange
from tg.draw import Draw, DrawGV
from tg.format import TextFormat, MtevalFormat
from tg.eval import mteval, mteval_lang, get_scores

log = logging.getLogger(__name__)

PREP_DIR = "prep"

def score_model(lang_pair, exp_dir, draw=True):
    """
    score using naive Bayes models, arrange translation, (optionally) draw
    graphs and calculate NIST/BLUE scores
    """
    # load graphs
    pkl_fname = join(PREP_DIR, lang_pair + "_graphs.pkl")
    log.info("loading graphs from " + pkl_fname)
    graph_list = cPickle.load(open(pkl_fname))
    
    # apply classifier
    models_fname = join(exp_dir, "models.hdf5")
    classifier = NearestCentroidClassifier(models_fname)
    source_lang = lang_pair.split("-")[0]
    scorer = ClassifierScore(classifier,
                             score_attr="centroid_score",
                             filter=filter_functions(source_lang))
    scorer(graph_list)
        
    # arrange 
    arrange = Arrange(score_attrs=["centroid_score", "freq_score"])
    arrange(graph_list)
    
    # draw
    if not os.path.exists(exp_dir): 
        os.makedirs(exp_dir)
    if draw:
        draw = Draw(drawer=DrawGV)
        draw(graph_list, out_format="pdf", out_dir=exp_dir,
             score_attrs=["centroid_score", "freq_score"])
    
    format = TextFormat()
    format(graph_list)
    txt_fname = join(exp_dir, "out.txt")
    format.write(txt_fname)
    
    # write translation output in Mteval format
    srclang, trglang = mteval_lang(lang_pair)
    format = MtevalFormat(srclang=srclang, trglang=trglang, sysid=exp_dir)
    format(graph_list)
    tst_fname = join(exp_dir, "out.tst")
    format.write(tst_fname)
    
    # calculate BLEU and NIST scores using mteval script
    score_fname = join(exp_dir, "out.scores")
    mteval(config["eval"]["presemt"][lang_pair]["lemma_ref_fname"],
           config["eval"]["presemt"][lang_pair]["src_fname"],
           tst_fname,
           score_fname)
    scores = get_scores(score_fname)
    log.info("scores for {0}: NIST = {1[0]}; BLEU = {1[1]}".format(
        exp_dir, scores))
    return scores
    




def run_all():
    language_pairs =  "de-en", "en-de"
    tab_fnames = ( join(config["private_data_dir"], "corpmod/de/en/de-en_ambig.tab"),
                   join(config["private_data_dir"], "corpmod/en/de/en-de_ambig.tab"))
    
    descriptor = {'names': ('lang_pair', 'NIST', 'BLUE'), 
                  'formats': ('S8', 'f4', 'f4')}
    results = np.zeros(9999, dtype=descriptor)
    exp_count = 0
    
    for lang_pair, tab_fname in zip(language_pairs, tab_fnames):
        target_lang = lang_pair.split("-")[1]
        samp_fname = target_lang + "_samples_subset_filtered.hdf5"
        graphs_pkl_fname = "prep/{}_graphs.pkl".format(lang_pair)
        
        results[exp_count] = lang_pair, 0, 0
        exp_dir = "exp_" + "_".join("{}={}".format(var, value) 
                                    for var, value in zip(results.dtype.names, 
                                                          results[exp_count])[:-2])
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        
        classifier = NearestCentroid()
        models_fname = join(exp_dir, "models.hdf5")
        
        ##builder = NearestCentroidModelBuilder(tab_fname, samp_fname, models_fname,
                                              ##classifier, graphs_pkl_fname=graphs_pkl_fname,
                                              ##feat_selector=SelectFpr(chi2)
                                              ##)
        ##builder.run()
        nist, blue = score_model(lang_pair, exp_dir)#, draw=False)
        results[exp_count]["NIST"] = nist 
        results[exp_count]["BLUE"] = blue
        
        exp_count += 1
            
    results = results[:exp_count]
    results.dump("centroid_results.pkl")
    
    



# for logging to stderr in utf-8 use:
set_default_log(level=logging.INFO)

import logging
logging.getLogger("model").setLevel(logging.DEBUG)    

run_all()    