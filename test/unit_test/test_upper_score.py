"""
unit test for upper scores
"""

import cPickle

from tg.config import config
from tg.upperscore import DictUpperScore
from tg.utils import set_default_log

from tg.format import TextFormat

# TODO:
# - check that result is indeed correct

class TestDictUpperScore:
    
    def test_dict_upper_score_de_en(self):
        graphs = cPickle.load(
            open(config["test_data_dir"] + "/graphs_sample_out_de-en.pkl"))
        maxscore = DictUpperScore(config["test_data_dir"] + 
                            "/lemma_sample_out_de-en.ref")
        maxscore(graphs)
        formatter = TextFormat(score_attr="dup_score")
        formatter(graphs)
        #formatter.write()
        
    def test_dict_upper_score_en_de(self):
        graphs = cPickle.load(open(config["test_data_dir"] + 
                                   "/graphs_sample_newstest2011-src.en.pkl"))
        maxscore = DictUpperScore(config["test_data_dir"] + 
                            "/lemma_sample_newstest2011-ref.de.sgm")
        maxscore(graphs)
        formatter = TextFormat(score_attr="dup_score")
        formatter(graphs)
        #formatter.write()
    
    
    
    