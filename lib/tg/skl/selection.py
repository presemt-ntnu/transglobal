"""
classes for feature selection
"""

# TODO
# - unit tests


import logging

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator
from sklearn.feature_selection.selector_mixin import SelectorMixin

log = logging.getLogger(__name__)



class MinCountFilter(BaseEstimator, SelectorMixin):
    """
    Filter vocabulary terms with a count below the minimum count
    """
    
    # SelectorMixin provides 
    #     transform(self, X, threshold=None) 
    #     fit_transform(self, X, y=None, **fit_params)
    # and requires
    #     self.feature_importances_
    #     self.threshold (optional)
    #     self.fit(X, **fit_params)
    #     self.transform(X)
    
    def __init__(self, min_count=5):
        self.min_count = min_count
        self.threshold = min_count
    
    def fit(self, X, y=None, **fit_params):
        if sp.issparse(X):
            # for sparse matrix, summation returns matrix
            self.feature_importances_ = np.asarray(X.sum(axis=0))[0]
        else:
            self.feature_importances_ = X.sum(axis=0)
        
        log.debug("MinCountFilter removed out {} of {} features".format(
            np.sum(self.feature_importances_ < self.threshold),
            X.shape[1]))
        
        return self
        
        
        
class MaxFreqFilter(BaseEstimator, SelectorMixin):
    """
    Filter vocabulary terms with a frequency above the maximum frequency
    threshold, where frequency is simply the mean value of a feature
    """
    
    def __init__(self, max_freq=.01):
        self.max_freq = max_freq
        self.threshold = 1
        
    def fit(self, X, y=None, **fit_params):
        if sp.issparse(X):
            # for sparse matrix, summation returns matrix
            means = np.asarray(X.mean(axis=0))[0]
        else:
            means = X.mean(axis=0)
        
        self.feature_importances_ = np.array(means <= self.max_freq, 
                                             dtype=np.int) 
        
        log.debug("MaxFreqFilter removed {} of {} features".format(
            X.shape[1] - self.feature_importances_.sum(),
            X.shape[1]))
        
        return self
        
        
        
        
