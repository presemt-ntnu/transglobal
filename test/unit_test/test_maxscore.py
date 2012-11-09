"""
unit test for maxscore
"""

import cPickle

from tg.config import config
from tg.maxscore import MaxScore
from tg.utils import set_default_log

from tg.format import TextFormat

# TODO:
# - check that result is indeed correct

class TestMaxScore:
    
    def test_maxscore_de_en(self):
        graphs = cPickle.load(
            open(config["test_data_dir"] + "/graphs_sample_out_de-en.pkl"))
        maxscore = MaxScore(config["test_data_dir"] + 
                            "/lemma_sample_out_de-en.ref")
        maxscore(graphs)
        formatter = TextFormat(score_attr="max_score")
        formatter(graphs)
        #formatter.write()
        
    def test_maxscore_en_de(self):
        graphs = cPickle.load(open(config["test_data_dir"] + 
                                   "/graphs_sample_newstest2011-src.en.pkl"))
        maxscore = MaxScore(config["test_data_dir"] + 
                            "/lemma_sample_newstest2011-ref.de.sgm")
        maxscore(graphs)
        formatter = TextFormat(score_attr="max_score")
        formatter(graphs)
        #formatter.write()
    
    
    
    