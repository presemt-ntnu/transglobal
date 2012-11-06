"""
Make test data

The created test data are part of the repos, so there is no need to run this,
unless the underlying code (e.g. for lemmatization) has changed and updated
versions of the test data are needed.
"""

from cPickle import dump
from codecs import open
from os.path import splitext
import logging


from tg.config import config
from tg.eval import lemmatize
from tg.annot import get_annotator
from tg.lookup import Lookup
from tg.transdict import TransDict
from tg.utils import set_default_log
from tg.draw import Draw
from tg.freqscore import FreqScore

log = logging.getLogger(__name__)   
set_default_log(level=logging.INFO)


class LookupKeepKeys(Lookup):
    """
    Hacked version of Lookup. As side effect, it creates a list of all
    succesfully looked up keys. This list can then be used to create a
    minimal dictionary containing only the keys required for a certain text.
    This miminal dict loads much faster and can be used in unit testing.
    """
    
    def __init__(self, *args, **kwargs):
        Lookup.__init__(self, *args, **kwargs)
        self.used_lempos_keys = set()
        self.used_lemma_keys = set()
        
    def _lookup_lempos_seq(self, lempos_seq):
        entries = Lookup._lookup_lempos_seq(self, lempos_seq)
        # keep track of all lempos and lemma keys
        for key, _ in entries:
            self.used_lempos_keys.add(key)
            lemma = key.rsplit("/",1)[0]
            self.used_lemma_keys.add(lemma)
        return entries
    
    def get_minimal_trans_dict(self):
        for key, values in self.dictionary._lemma_dict.items():
            if key not in self.used_lemma_keys:
                del self.dictionary._lemma_dict[key]
            else:
                for lempos in values:
                    self.used_lempos_keys.add(lempos)
                
        for key in self.dictionary._lempos_dict.keys():
            if key not in self.used_lempos_keys:
                del self.dictionary._lempos_dict[key]
                
            
        return self.dictionary


def lemmatize_reference():
    lemmatize("sample_out_de-en.ref", "en", 
              "lemma_sample_out_de-en.ref")
    lemmatize("sample_newstest2011-ref.de.sgm", "de",
              "lemma_sample_newstest2011-ref.de.sgm")
    
def make_graphs():
    for lang_pair, src_fname in [ ("en-de", "sample_newstest2011-src.en.sgm"),
                                  ("de-en", "sample_out_de-en.src") ]:
        source_lang, target_lang = lang_pair.split("-")
        root_fname = splitext(src_fname)[0]
        
        # annotate
        annotator = get_annotator(source_lang)
        graphs = annotator.annot_xml_file(src_fname)    
        
        # lookup
        dict_fname = config["dict"][lang_pair]["pkl_fname"]
        trans_dict = TransDict.load(dict_fname)
        lookup = LookupKeepKeys(trans_dict)
        lookup(graphs)
        
        #  write pickle of minimal translation dict
        min_dict = lookup.get_minimal_trans_dict()
        min_dict_fname = "dict_" + root_fname + ".pkl"
        dump(min_dict, open(min_dict_fname, "wb"))
        
        # score most frequent translation
        counts_fname = config["count"]["lemma"][target_lang]["pkl_fname"]
        freq_score = FreqScore(counts_fname)
        freq_score(graphs)
        
        # draw graphs
        draw = Draw()
        draw(graphs, out_format="pdf", best_score_attr="freq_score", 
             out_dir="_draw_" + lang_pair)
        
        # save graphs
        graphs_fname = "graphs_" + root_fname + ".pkl"
        dump(graphs, open(graphs_fname, "wb"))
    
    

lemmatize_reference()    
make_graphs()
