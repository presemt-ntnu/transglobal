"""
classes for feature selection
"""


import logging

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator
from sklearn.utils import atleast2d_or_csr

try:
    from sklearn.feature_selection.from_model import _LearntSelectorMixin
except ImportError:
    # sklearn<0.13
    from sklearn.feature_selection.selector_mixin import SelectorMixin as\
         _LearntSelectorMixin

log = logging.getLogger(__name__)



class MinCountFilter(BaseEstimator, _LearntSelectorMixin):
    """
    Filter out all features (vocabulary terms) with an instance (document)
    count below the minimum count threshold.
    """
    
    # _LearntSelectorMixin provides 
    #     transform(self, X, threshold=None) 
    #     fit_transform(self, X, y=None, **fit_params)
    # and requires
    #     self.feature_importances_
    #     self.threshold (optional)
    #     self.fit(X, **fit_params)
    
    def __init__(self, min_count=5):
        self.min_count = min_count
        self.threshold = min_count
    
    def fit(self, X, y=None, **fit_params):
        X = atleast2d_or_csr(X)
        if sp.issparse(X):
            # Indices attrib of sparse matrix in csr format contains column
            # numbers of non-zero elements. Counting the number of
            # occurrences of each value in this array therefore gives the
            # number of non-zero elements per column. The minlength argument
            # guarantees even zero-count columns at the end of the matrix are
            # included.
            assert X.format == "csr"
            self.feature_importances_ = np.bincount(X.indices, minlength=X.shape[1])
        else:
            self.feature_importances_ = (X > 0).sum(axis=0)
        
        log.debug("MinCountFilter removed out {} of {} features".format(
            np.sum(self.feature_importances_ < self.threshold),
            X.shape[1]))
        
        return self
        
        
        
class MaxFreqFilter(BaseEstimator, _LearntSelectorMixin):
    """
    Filter out features (vocabulary terms) with a instance (document)
    frequency above the maximum frequency threshold.
    """
    
    def __init__(self, max_freq=.05):
        self.max_freq = max_freq
        self.threshold = 1
        
    def fit(self, X, y=None, **fit_params):        
        X = atleast2d_or_csr(X)        
        if sp.issparse(X):
            # Indices attrib of sparse matrix in csr format contains column
            # numbers of non-zero elements. Counting the number of
            # occurrences of each value in this array therefore gives the
            # number of non-zero elements per column. The minlength argument
            # guarantees even zero-count columns at the end of the matrix are
            # included.
            counts = np.bincount(X.indices, minlength=X.shape[1])
        else:
            counts = (X > 0).sum(axis=0)
            
        max_count = self.max_freq * X.shape[0]
        # All features with a count above max_count get become "0" whereas
        # all others become "1". Using a threshold of "1", transform will
        # then select the featurs with a count below or equal to max_count.
        self.feature_importances_ = (counts <= max_count).astype("i")
        
        log.debug("MaxFreqFilter removed {} of {} features".format(
            X.shape[1] - self.feature_importances_.sum(),
            X.shape[1]))
        
        return self
