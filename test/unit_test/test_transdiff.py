"""
unit tests for transdiff
"""

from tg.config import config
from tg.transdiff import trans_diff

class TestTransdiff:
            
    def test_transdiff(self):
        trans_diff(
            config["test_data_dir"] + "/graphs_sample_out_de-en.pkl",
            ["rand_score", "freq_score"])
        
    def test_transdiff_with_ref(self):
        trans_diff(
            config["test_data_dir"] + "/graphs_sample_out_de-en.pkl",
            ["rand_score", "freq_score"],
            config["test_data_dir"] + "/sample_out_de-en.ref")
        
        