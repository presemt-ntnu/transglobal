"""
test random scoring
"""

from cPickle import load

from tg.randscore import RandProb
from tg.config import config



class TestRandProb:
    
    @classmethod
    def setup_class(cls):
        # load graphs
        graphs_fname = config["test_data_dir"] + "/graphs_sample_out_de-en.pkl"
        cls.graph_list = load(open(graphs_fname))
        
    def test_random(self):
        rnd = RandProb()
        rnd(self.graph_list)
        
        for graph in self.graph_list:
            for u, v, d in graph.trans_edges_iter():
                assert d["rand_score"]
