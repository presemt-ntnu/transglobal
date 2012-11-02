"""
unit test for maxscore
"""

import cPickle
import unittest

from tg.maxscore import MaxScore
from tg.utils import set_default_log

from tg.format import TextFormat


class TestMaxScore(unittest.TestCase):
    
    def test_maxscore_de_en(self):
        graphs = cPickle.load(
            open("../data/graphs_sample_out_de-en.src.pkl"))
        maxscore = MaxScore("../data/lemma_sample_out_de-en.ref")
        maxscore(graphs)
        formatter = TextFormat(score_attr="max_score")
        formatter(graphs)
        #formatter.write()
        
    def test_maxscore_en_de(self):
        graphs = cPickle.load(
            open("../data/graphs_sample_newstest2011-src.en.pkl"))
        maxscore = MaxScore("../data/lemma_sample_newstest2011-ref.de.sgm")
        maxscore(graphs)
        formatter = TextFormat(score_attr="max_score")
        formatter(graphs)
        #formatter.write()
    
    
if __name__ == '__main__':
    #set_default_log()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMaxScore)
    unittest.TextTestRunner(verbosity=2).run(suite)    
    
    