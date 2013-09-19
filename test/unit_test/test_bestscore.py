"""
test best scoring
"""

from cPickle import load

from tg.bestscore import BestScorer
from tg.config import config



class TestBestScorer:
    
    @classmethod
    def setup_class(cls):
        # load graphs
        graphs_fname = config["test_data_dir"] + "/graphs_sample_out_de-en.pkl"
        cls.graph_list = load(open(graphs_fname))
        
    def test_best_score_1(self):
        self.clear_scores(self.graph_list)
        scorer = BestScorer(base_score_attrs=["freq_score","dup_score"])
        scorer(self.graph_list)
        self.check_scores(self.graph_list, "freq_score")
        
    def test_best_score_2(self):
        self.clear_scores(self.graph_list)
        scorer = BestScorer(base_score_attrs=["dup_score", "freq_score"])
        scorer(self.graph_list)
        self.check_scores(self.graph_list, "dup_score")
                
    def clear_scores(self, graphs):
        for graph in graphs:
            for u, v, data in graph.trans_edges_iter():
                try:
                    del data["best_score"]  
                except KeyError:
                    pass
                
    def check_scores(self, graphs, other_attr):
        for graph in graphs:
            for u, v, data in graph.trans_edges_iter():
                # FIXME: hypernodes should be scored too
                if not graph.is_hyper_source_node(u):
                    assert data["best_score"] == data[other_attr]  
