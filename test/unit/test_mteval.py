"""
unit tests for mteval
"""

from nose.tools import assert_almost_equal

from tg.config import config
from tg.mteval import ( parse_document_scores,
                        parse_segment_scores,
                        parse_total_scores )

score_fname = config["test_data_dir"] + "/mft_wmt10_de-en.scores"


class TestMteval:
    
    def test_parse_total_scores(self):
        scores = parse_total_scores(score_fname)
        assert scores == ( "most frequent translation",
                           4.8555,
                           0.1093 )
    
    def test_parse_document_scores(self):
        scores = parse_document_scores(score_fname)
        assert scores.shape[0] == 119
        assert scores["segments"].sum() == 2489
        
    def test_parse_segment_scores(self):
        scores = parse_segment_scores(score_fname)
        assert scores.shape[0] == 2489
        assert_almost_equal(scores[0]["NIST"], 1.1990) 
        assert_almost_equal(scores[0]["BLEU"], 0.0375) 
        