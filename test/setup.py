#!/usr/bin/env python

"""
setup local data for testing
"""

from os import getenv
from os.path import join

from tg.config import config

test_data_dir = join(config["TG_BASE_DIR"], "test/data")

test_local_dir = join(config["TG_BASE_DIR"], "_local/data")

annot_graphs_en_pkl_fname = join(test_local_dir, "annot_graphs_en.pkl")
graphs_en_de_pkl_fname = join(test_local_dir, "graphs_en-de.pkl")
graphs_en_rnd_pkl_fname = join(test_local_dir, "graphs_en-de_rnd.pkl")


def make_graphs_en():
    from cPickle import dump
    from codecs import open
    
    from tg.annot import TreeTaggerEnglish
    from tg.transdict import TransDict, DictAdaptor
    from tg.lookup import Lookup
    from tg.randscore import RandProb
 
    # annotate
    log.info("using TreeTagger for English")
    annotator = TreeTaggerEnglish()  
    
    text_fname = test_data_dir + "/sample_en_1.txt"
    log.info("annotating text from: " + text_fname)
    text = open(text_fname, encoding="utf-8").read()
    graph_list = annotator(text)

    log.info("writing pickled graph to " + annot_graphs_en_pkl_fname)
    dump(graph_list, open(annot_graphs_en_pkl_fname, "wb"))
    
    # lookup - requires pickled dictionary
    en_de_dict = DictAdaptor(config["en-de_dict_pkl"],
                             config["en-de_posmap"])
    lookup = Lookup(en_de_dict)
    lookup(graph_list)
    log.info("writing pickled graph to " + graphs_en_de_pkl_fname)
    dump(graph_list, open(graphs_en_de_pkl_fname, "wb"))
    
    # random scores
    rnd = RandProb()
    rnd(graph_list)
    log.info("writing graphs with random probs to " + graphs_en_rnd_pkl_fname)
    dump(graph_list, open(graphs_en_rnd_pkl_fname, "wb"))
    
    

    
if __name__ == "__main__":
    import logging as log
    log.basicConfig(level=log.INFO)
    
    from os.path import exists
    from os import mkdir
    
    if not exists(test_local_dir):
        log.info("Creating test local dir: " + test_local_dir)
        mkdir(test_local_dir)
        
    make_graphs_en()
        
    