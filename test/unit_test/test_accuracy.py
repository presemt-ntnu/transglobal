"""
test accuracy score
"""

import cPickle

from nose.tools import assert_almost_equal

from tg.config import config
from tg.accuracy import accuracy


def test_accuracy():
    graphs_fname = config["test_data_dir"] + "/graphs_sample_out_de-en.pkl"
    graphs = cPickle.load(open(graphs_fname))
    ref_fname = config["test_data_dir"] + "/lemma_sample_out_de-en.ref"
    result = accuracy(graphs[:1], ref_fname, "freq_score")
    assert result.correct == 5 
    assert result.incorrect == 1  
    assert result.ignored == 1
    assert_almost_equal(result.score, 0.8333333)


        