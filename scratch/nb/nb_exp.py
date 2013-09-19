#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
running WTD experiments with Naive bayes models
"""

import cPickle
import logging
import os
from os.path import join
import xml.etree.cElementTree as et

from tg.config import config
from tg.annot import TreeTaggerEnglish, TreeTaggerGerman
from tg.transdict import TransDict
from tg.lookup import Lookup
from tg.freqscore import FreqScorer
from tg.draw import Draw, DrawGV
from tg.arrange import Arrange
from tg.format import TextFormat, MtevalFormat
from tg.eval import mteval, mteval_lang, get_scores
from tg.classify import NaiveBayesClassifier
from tg.nbscore import NBScore

log = logging.getLogger(__name__)

PREP_DIR = "prep"



def prepare(lang_pair):
    """
    extract input text, create annotated graphs, lookup translation
    candidates, perform frequency scoring and save pickled graphs to file
    """
    source_lang, target_lang = lang_pair.split("-")
    
    # get text from input source
    xml_tree = et.ElementTree(file=config["eval"]["presemt"][lang_pair]["src_fname"])
    text = " ".join(seg.text.strip() for seg in  xml_tree.iter("seg"))

    # annotate
    if source_lang == "en":
        annotator = TreeTaggerEnglish()
    elif source_lang == "de":
        annotator = TreeTaggerGerman()
    else:
        raise ValueError("unknown source language: " + source_lang)
    graph_list = annotator(text)

    # lookup
    trans_dict = TransDict.load(config["dict"][lang_pair]["pkl_fname"])
    lookup = Lookup(trans_dict)
    lookup(graph_list)

    # frequency scoring
    freq_score = FreqScorer(config["count"]["lemma"][target_lang]["pkl_fname"])
    freq_score(graph_list)

    # save
    if not os.path.exists(PREP_DIR):
        os.makedirs(PREP_DIR)
    pkl_fname = join(PREP_DIR, lang_pair + "_graphs.pkl")
    log.info("saving graphs to " + pkl_fname)
    cPickle.dump(graph_list, 
                 open(pkl_fname, "wb"),
                 protocol=cPickle.HIGHEST_PROTOCOL)



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
    models_fname = join(exp_dir, "nb_models.hdf5")
    classifier = NaiveBayesClassifier(models_fname)
    nbscorer = NBScore(classifier)
    nbscorer(graph_list)
        
    # arrange 
    arrange = Arrange(score_attrs=["nb_score", "freq_score"])
    arrange(graph_list)
    
    # draw
    if not os.path.exists(exp_dir): 
        os.makedirs(exp_dir)
    if draw:
        draw = Draw(drawer=DrawGV)
        draw(graph_list, out_format="pdf", out_dir=exp_dir,
             score_attrs=["nb_score", "freq_score"])
    
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
    



