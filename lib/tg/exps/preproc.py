"""
preprocessing of experimental data
"""

import logging 
log = logging.getLogger(__name__)

import cPickle
import os

from tg.config import config
from tg.annot import get_annotator
from tg.transdict import TransDict
from tg.lookup import Lookup
from tg.freqscore import FreqScorer
from tg.upperscore import DictUpperScorer



def preprocess(data_set, lang_pair):
    source_lang, target_lang = lang_pair.split("-")
    graphs_fname = config["eval"][data_set][lang_pair]["graphs_fname"]
    out_dir = os.path.dirname(graphs_fname)
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
    freq_score = FreqScorer(config["count"]["lemma"][target_lang]["pkl_fname"])
    freq_score(graph_list)
    
    # approximated max scores  
    lemma_ref_fname = \
        config["eval"][data_set][lang_pair]["lemma_ref_fname"]
    maxscore = DictUpperScorer(lemma_ref_fname)
    maxscore(graph_list)
    
    # save graphs
    log.info("saving preprocessed graphs to " + graphs_fname)
    cPickle.dump(graph_list, open(graphs_fname, "wb"))
    