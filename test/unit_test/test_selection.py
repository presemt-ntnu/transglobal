"""
test feature selection
"""

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

import scipy.sparse as sp

from tg.skl.selection import MaxFreqFilter, MinCountFilter


class TestMinCountFilter:
        
    def test_min_count_filter(self):
        X = np.array([[1.0, 0.0], 
                      [2.0, 1.0], 
                      [3.0, 0.0], 
                      [2.0, 1.0], 
                      [4.0, 1.0], 
                      [6.0, 1.0]])   
        min_count = 5
        filt = MinCountFilter(min_count=min_count)
        Xt = filt.fit_transform(X)
        assert all(X[:, X.sum(axis=0) >= min_count] == Xt)
        
    def test_min_count_filter_sparse(self):
        X = sp.csr_matrix([[1.0, 0.0], 
                           [2.0, 1.0], 
                           [3.0, 0.0], 
                           [2.0, 1.0], 
                           [4.0, 1.0], 
                           [6.0, 1.0]])        
        filt = MinCountFilter(min_count=5)
        Xt = filt.fit_transform(X)
        assert Xt.shape == (6,1)


class TestMaxFreqFilter:
        
    def test_max_freq_filter(self):
        X = np.array([[1.0, 0.0], 
                      [0.0, 0.0], 
                      [1.0, 0.0], 
                      [1.0, 1.0], 
                      [1.0, 1.0], 
                      [0.0, 0.0]])   
        max_freq = 0.5
        filt = MaxFreqFilter(max_freq=max_freq)
        Xt = filt.fit_transform(X)
        assert all(X[:, X.mean(axis=0) < max_freq] == Xt) 
        
    def test_max_freq_filter_sparse(self):
        X = sp.csr_matrix([[1.0, 0.0], 
                           [0.0, 0.0], 
                           [1.0, 0.0], 
                           [1.0, 1.0], 
                           [1.0, 1.0], 
                           [0.0, 0.0]])        
        filt = MaxFreqFilter(max_freq=0.5)
        X = filt.fit_transform(X)
        assert X.shape == (6,1)
        