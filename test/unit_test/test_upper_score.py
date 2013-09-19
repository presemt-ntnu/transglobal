"""
unit test for upper scores
"""

import cPickle

from tg.config import config
from tg.upperscore import DictUpperScorer
from tg.utils import set_default_log

from tg.format import TextFormat

#import logging
#set_default_log(level=logging.DEBUG)


class TestDictUpperScorer:
    
    def test_dict_upper_score_de_en(self):
        graphs = cPickle.load(
            open(config["test_data_dir"] + "/graphs_sample_out_de-en.pkl"))
        self.clear_scores(graphs)
        ref_fname = config["test_data_dir"] + "/lemma_sample_out_de-en.ref"
        scorer = DictUpperScorer(ref_fname)
        scorer(graphs)
        self.check_scores(graphs)
        
    def test_dict_upper_score_en_de(self):
        graphs = cPickle.load(open(config["test_data_dir"] + 
                                   "/graphs_sample_newstest2011-src.en.pkl"))
        self.clear_scores(graphs)
        scorer = DictUpperScorer(config["test_data_dir"] + 
                                "/lemma_sample_newstest2011-ref.de.sgm")
        scorer(graphs)
        self.check_scores(graphs)
        
    def clear_scores(self, graphs):
        for graph in graphs:
            for u, v, data in graph.trans_edges_iter():
                try:
                    del data["dup_score"]  
                except KeyError:
                    pass
                
    def check_scores(self, graphs):
        for graph in graphs:
            for u, v, data in graph.trans_edges_iter():
                # FIXME: hypernodes should be scored too
                if not graph.is_hyper_source_node(u):
                    assert 0.0 <= data["dup_score"] <= 1.0  
        
        
    
    
    
    