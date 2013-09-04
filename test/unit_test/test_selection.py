"""
test feature selection
"""

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

import scipy.sparse as sp

from tg.skl.selection import MaxFreqFilter, MinCountFilter


class TestMinCountFilter:
        
    def test_min_count_filter(self):
        X = np.array([[1.0, 0.0, 0.0], 
                      [2.0, 1.0, 0.0], 
                      [3.0, 0.0, 0.0], 
                      [2.0, 1.0, 0.0], 
                      [4.0, 1.0, 0.0], 
                      [6.0, 9.0, 0.0]])    
        filt = MinCountFilter(min_count=5)
        Xt = filt.fit_transform(X)
        # expected X
        Xe = X[:,0][np.newaxis].T
        assert_array_equal(Xt, Xe)
        
    def test_min_count_filter_sparse(self):
        X = sp.csr_matrix([[1.0, 0.0, 0.0], 
                           [2.0, 1.0, 0.0], 
                           [3.0, 0.0, 0.0], 
                           [2.0, 1.0, 0.0], 
                           [4.0, 1.0, 0.0], 
                           [6.0, 9.0, 0.0]])    
        filt = MinCountFilter(min_count=5)
        Xt = filt.fit_transform(X)
        # expected X
        Xe = X[:,0]
        assert Xt.shape == (6,1)
        # __eq__ is not properly implemented for sparse matrices
        # so apply this trick
        assert (Xt - Xe).nnz == 0


class TestMaxFreqFilter:
        
    def test_max_freq_filter(self):
        X = np.array([[1.0, 0.0, 0.0], 
                      [0.0, 0.0, 0.0], 
                      [1.0, 9.0, 0.0], 
                      [1.0, 9.0, 0.0], 
                      [1.0, 9.0, 0.0], 
                      [0.0, 0.0, 0.0]])   
        max_freq = 0.5
        filt = MaxFreqFilter(max_freq=max_freq)
        Xt = filt.fit_transform(X)
        # expected X
        Xe = X[:,1:]
        assert_array_equal(Xt, Xe)
        
    def test_max_freq_filter_sparse(self):
        X = sp.csr_matrix([[1.0, 0.0, 0.0], 
                           [0.0, 0.0, 0.0], 
                           [1.0, 9.0, 0.0], 
                           [1.0, 9.0, 0.0], 
                           [1.0, 9.0, 0.0], 
                           [0.0, 0.0, 0.0]])        
        filt = MaxFreqFilter(max_freq=0.5)
        Xt = filt.fit_transform(X)
        # expected X
        Xe = X[:,1:]
        assert Xt.shape == (6,2)
        # __eq__ is not properly implemented for sparse matrices
        # so apply this trick
        assert (Xt - Xe).nnz == 0
        