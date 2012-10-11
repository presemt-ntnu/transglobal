"""
test nearest centroid classifier with cosine similarity
"""

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

import scipy.sparse as sp

from tg.skl.centroid import CosNearestCentroid


class TestCosNearestCentroid:
    
    @classmethod
    def setup_class(cls):
        cls.X = np.array([[1.0, 2.0], 
                          [2.0, 4.0], 
                          [3.0, 6.0], 
                          [2.0, 1.0], 
                          [4.0, 2.0], 
                          [6.0, 3.0]])
        cls.y = np.array([1, 1, 1, 2, 2, 2])      
    
    def test_predict(self):
        clf = CosNearestCentroid()
        clf.fit(self.X, self.y)        
        assert_array_equal(clf.predict(self.X), self.y)   
    
    def test_predict_sparse(self):
        clf = CosNearestCentroid()
        X_sp = sp.csr_matrix(self.X)
        clf.fit(X_sp, self.y)        
        assert_array_equal(clf.predict(X_sp), self.y) 
        
    def test_predict_proba(self):
        clf = CosNearestCentroid()
        clf.fit(self.X, self.y)
        probs = clf.predict_proba(self.X)
        # check that probabilities sum to one
        prob_sums = probs.sum(axis=1)
        assert_array_equal(prob_sums, np.ones_like(prob_sums))
        
    def test_predict_proba_sparse(self):
        clf = CosNearestCentroid()
        X_sp = sp.csr_matrix(self.X)
        clf.fit(X_sp, self.y)
        probs = clf.predict_proba(X_sp)
        prob_sums = probs.sum(axis=1)
        assert_array_equal(prob_sums, np.ones_like(prob_sums))
        

if __name__ == "__main__":
    import nose, sys
    sys.argv.append("-v")
    nose.run(defaultTest=__name__)