"""
test nearest centroid classifier with cosine similarity
"""

import numpy as np
from numpy.testing import ( assert_almost_equal, assert_array_equal,
                            assert_array_almost_equal )

import scipy.sparse as sp

from tg.skl.centroid import NearestCentroidProb


class TestNearestCentroidPred:
    
    @classmethod
    def setup_class(cls):
        X = np.array([[1.0, 2.0], 
                      [2.0, 4.0], 
                      [3.0, 6.0], 
                      [2.0, 1.0], 
                      [4.0, 2.0], 
                      [6.0, 3.0]])
        # turn into sparse matrix
        cls.X = sp.csr_matrix(X)
        cls.y = np.array([1, 1, 1, 2, 2, 2])   
        
    def test_predict_proba(self):
        """
        test predicting probabilities with cosine similarit on sparse matrix
        """
        clf = NearestCentroidProb(metric="cosine")
        clf.fit(self.X, self.y)
        probs = clf.predict_proba(self.X)
        
        # check that probabilities sum to one
        prob_sums = probs.sum(axis=1)
        assert_array_equal(prob_sums, np.ones_like(prob_sums))

        # check for expected probs provided that self.centroids_ is
        # array([[ 2.,  4.],
        #        [ 4.,  2.]])
        expected = np.array([[1.0, 0.0], 
                             [1.0, 0.0], 
                             [1.0, 0.0], 
                             [0.0, 1.0], 
                             [0.0, 1.0], 
                             [0.0, 1.0]])
        assert_array_almost_equal(probs, expected)
        

