"""
Make test data

The created test data are part of the repos, so there is no need to run this,
unless the underlying code (e.g. for lemmatization) has changed and updated
versions of the test data are needed.
"""

from cPickle import dump
from codecs import open
import logging


from tg.config import config
from tg.eval import lemmatize
from tg.annot import TreeTaggerEnglish, TreeTaggerGerman
from tg.lookup import Lookup
from tg.transdict import TransDict
from tg.utils import set_default_log

log = logging.getLogger(__name__)   
set_default_log(level=logging.INFO)


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


def lemmatize_reference():
    lemmatize("sample_out_de-en.ref", "en", 
              "lemma_sample_out_de-en.ref")
    lemmatize("sample_newstest2011-ref.de.sgm", "de",
              "lemma_sample_newstest2011-ref.de.sgm")
    
def make_graphs():
    params = [
        (TreeTaggerEnglish(), 
         "sample_newstest2011-src.en.sgm",
         config["dict"]["en-de"]["pkl_fname"],
         "dict_en-de_min.pkl",
         "graphs_sample_newstest2011-src.en.pkl"),
        (TreeTaggerGerman(), 
         "sample_out_de-en.src",
         config["dict"]["de-en"]["pkl_fname"],
         "dict_de-en_min.pkl",
         "graphs_sample_out_de-en.src.pkl"),
         ]
    
    for annotator, src_fname, dict_fname, min_dict_fname, graphs_fname in params:
        # annotate
        graphs = annotator.annot_xml_file(src_fname)    
        
        # lookup
        trans_dict = TransDict.load(dict_fname)
        lookup = LookupKeepKeys(trans_dict)
        lookup(graphs)
        
        #  write pickle of minimal translation dict
        min_dict = lookup.get_minimal_trans_dict()
        dump(min_dict, open(min_dict_fname, "wb"))
        
        # save graphs
        dump(graphs, open(graphs_fname, "wb"))
    
    

lemmatize_reference()    
make_graphs()
