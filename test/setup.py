#!/usr/bin/env python

"""
setup local data for testing
"""

from cPickle import dump
from codecs import open
from os import getenv
from os.path import join

from tg.config import config
from tg.annot import TreeTaggerEnglish
from tg.transdict import TransDict, DictAdaptor
from tg.lookup import Lookup
from tg.randscore import RandProb


test_data_dir = join(config["TG_BASE_DIR"], "test/data")

test_local_dir = join(config["TG_BASE_DIR"], "test/_local")

annot_graphs_en_pkl_fname = join(test_local_dir, "annot_graphs_en.pkl")
graphs_en_de_pkl_fname = join(test_local_dir, "graphs_en-de.pkl")
graphs_en_rnd_pkl_fname = join(test_local_dir, "graphs_en-de_rnd.pkl")
en_de_dict_pkl_fname = join(test_local_dir, "dict_en-de_min.pkl")


class LookupKeepKeys(Lookup):
    """
    Hacked version of Lookup. As side effect, it creates a list of all
    succesfully looked up keys. This list can then be used to create a
    minimal dictionary containing only the keys required for a certain text.
    This miminal loads much faster and can be used in unit testing.
    """
    
    def __init__(self, *args, **kwargs):
        Lookup.__init__(self, *args, **kwargs)
        self.keys = []
        
    def _lookup(self, key):
        # if lookup fails, a KeyError will be raised and the key will not be
        # added to list of keys
        value = Lookup._lookup(self, key)
        self.keys.append(key)
        return value
    
    def get_minimal_trans_dict(self):
        return TransDict( (key, self.dictionary[key]) 
                          for key in self.keys )



def setup_en():
 
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
    en_de_dict = DictAdaptor(config["dict"]["en-de"]["pkl_fname"],
                             config["dict"]["en-de"]["posmap_fname"])
    lookup = LookupKeepKeys(en_de_dict)
    lookup(graph_list)
    log.info("writing pickled graph to " + graphs_en_de_pkl_fname)
    dump(graph_list, open(graphs_en_de_pkl_fname, "wb"))
    
    # write pickle of minimal translation dict
    min_dict = lookup.get_minimal_trans_dict()
    log.info("writing minimal pickled translation dict to " + en_de_dict_pkl_fname)
    dump(min_dict, open(en_de_dict_pkl_fname, "wb"))
    
    # random scores
    rnd = RandProb()
    rnd(graph_list)
    log.info("writing graphs with random probs to " + graphs_en_rnd_pkl_fname)
    dump(graph_list, open(graphs_en_rnd_pkl_fname, "wb"))
    
    

if __name__ == "__main__":
    import logging as log
    log.basicConfig(level=log.INFO)
    log.getLogger().setLevel(log.INFO)
    
    from os.path import exists
    from os import mkdir
    
    if not exists(test_local_dir):
        log.info("Creating test local dir: " + test_local_dir)
        mkdir(test_local_dir)
        
    setup_en()
        
    