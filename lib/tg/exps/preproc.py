"""
preprocessing of experimental data
"""

import codecs
import logging
import cPickle
import os
import xml.etree.cElementTree as et

from tg.config import config
from tg.graphproc import GraphProces
from tg.annot import get_annotator
from tg.transdict import TransDict, TransDictGreek
from tg.lookup import Lookup
from tg.freqscore import FreqScore


def preprocess(data_set, lang_pair, 
               base_dir=""):
    source_lang, target_lang = lang_pair.split("-")
    exp_name = "mft_{}_{}".format(data_set, lang_pair)
    out_dir = os.path.join(base_dir, "_" + exp_name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    # annotate
    annotator = get_annotator(source_lang)
    graph_list = annotator.annot_xml_file(
        config["eval"][data_set][lang_pair]["src_fname"])
    
    # lookup translations
    dict_fname = TransDict.load(config["dict"][lang_pair]["pkl_fname"])
    lookup = Lookup(dict_fname)
    lookup(graph_list)
    
    # score most frequent translation
    freq_score = FreqScore(config["count"]["lemma"][target_lang]["pkl_fname"])
    freq_score(graph_list)
    
    # save graphs
    pkl_fname = os.path.join(out_dir, exp_name + ".pkl")
    cPickle.dump(graph_list, open(pkl_fname, "wb"))
    
    return out_dir, exp_name, graph_list
    