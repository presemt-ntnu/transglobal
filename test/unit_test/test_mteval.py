"""
unit tests for mteval
"""

from nose.tools import assert_almost_equal

from tg.config import config
from tg.mteval import ( parse_document_scores,
                        parse_segment_scores,
                        parse_total_scores,
                        read_ref_trans,
                        read_ref_trans_counts )

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
        
    def test_read_ref_trans(self):
        refset = read_ref_trans(
            config["test_data_dir"] + "/lemma_sample_out_de-en.ref")
        # check doc ids
        assert refset.keys() == ["test"]
        # check seg ids
        assert refset["test"].keys() == ['1', '2', '3', '4', '5']
        # check translations
        assert refset["test"]["5"] == [
            'all other aspect be secondary .', 
            'the rest be secondary .', 
            'All the rest be secondary .', 
            'All the rest be secondary .', 
            'the rest be secondary .']    
        
    def test_read_ref_trans_counts(self):
        refset = read_ref_trans_counts(
            config["test_data_dir"] + "/lemma_sample_out_de-en.ref") 
        for lemma, count in [("the", 6),
                             ("union", 5),
                             ("side", 1)]:
            assert refset["test"]['1'][lemma] == count
        