"""
test random scoring
"""

from cPickle import load

from tg.randscore import RandProb

from setup import graphs_en_de_pkl_fname


class TestRandProb:
    
    @classmethod
    def setup_class(cls):
        # load graphs
        cls.graph_list = load(open(graphs_en_de_pkl_fname))
        
    def test_random(self):
        rnd = RandProb()
        rnd(self.graph_list)
        
        
                            
if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__name__)        