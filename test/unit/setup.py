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
from tg.transdict import TransDict
from tg.lookup import Lookup
from tg.randscore import RandProb


test_data_dir = join(config["TG_BASE_DIR"], "test/unit/data")
integration_data_dir = join(config["TG_BASE_DIR"], "test/integration/data")

test_local_dir = join(config["TG_BASE_DIR"], "test/unit/_local")

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
        self.used_keys = set()
        
    def _lookup_lempos_seq(self, lempos_seq):
        entries = Lookup._lookup_lempos_seq(self, lempos_seq)
        # keep track of all lempos and lemma keys
        for key, _ in entries:
            self.used_keys.add(key)
        return entries
    
    def get_minimal_trans_dict(self):
        for key in self.dictionary._lempos_dict.keys():
            if key not in self.used_keys:
                del self.dictionary._lempos_dict[key]
                
        for key in self.dictionary._lemma_dict.keys():
            if key not in self.used_keys:
                del self.dictionary._lemma_dict[key]
            
        return self.dictionary




def setup_en():
 
    # annotate
    log.info("using TreeTagger for English")
    annotator = TreeTaggerEnglish()  
    
    text_fname = integration_data_dir + "/sample_en_1.txt"
    log.info("annotating text from: " + text_fname)
    text = open(text_fname, encoding="utf-8").read()
    graph_list = annotator.annot_text(text)

    log.info("writing pickled graph to " + annot_graphs_en_pkl_fname)
    dump(graph_list, open(annot_graphs_en_pkl_fname, "wb"))
    
    # lookup - requires pickled dictionary
    pkl_fname = config["dict"]["en-de"]["pkl_fname"]
    en_de_dict = TransDict.load(pkl_fname)
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
        
    