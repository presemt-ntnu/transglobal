
import numpy as np

from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import normalize
from sklearn.utils.validation import atleast2d_or_csr


class CosNearestCentroid(NearestCentroid):
    
    """
    Variant of nearest centroid classifier that uses cosine as similarity
    metric and provides a predict_proba() method
    
    Assumes that both training pass to "fit" and test data passed to
    "predict" is normalized, e.g. by TfidfTransformer(norm=l2). Centroids are
    also normalized. This means that computation of cosine reduces to dot
    product. Works for sparse matrices. 
    """
    
    def __init__(self, norm=True):
        # Just set metric to "cosine" 
        # Shrinking is not supported for sparse matrices anyway.
        NearestCentroid.__init__(self, metric="cosine")
        self.norm = norm
     
    def fit(self, X, y): 
        if self.norm:
            normalize(X, copy=False)
        NearestCentroid.fit(self, X, y)
        normalize(self.centroids_, copy=False)
        return self
    
    def predict(self, X):
        # Problem: pairwise.pairwise_distances called in
        # NearestCentroid.predict does not support cosine similarity for
        # sparse matrices
        X = atleast2d_or_csr(X)
        if not hasattr(self, "centroids_"):
            raise AttributeError("Model has not been trained yet.")
        if self.norm:
            normalize(X, copy=False)
        return self.classes_[X.dot(self.centroids_.T).argmax(axis=1)]
    
    def predict_proba(self, X):
        # NB assumes X is normalized
        # This is evidently silly, as normalizing the cosine similarities
        # over all classes does not turn them into probabilities. However, we
        # need this method in tg.classify.TranslationClassifier._predict()
        X = atleast2d_or_csr(X)
        if not hasattr(self, "centroids_"):
            raise AttributeError("Model has not been trained yet.")
        if self.norm:
            normalize(X, copy=False)
        cos_scores = X.dot(self.centroids_.T) 
        norms = cos_scores.sum(axis=1).reshape(-1,1)
        return cos_scores / norms
        
