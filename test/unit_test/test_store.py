"""
test storage of estimators and disambiguators in hdf5 file
"""

import tempfile

import numpy as np
from numpy.testing import assert_array_equal

from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import NearestCentroid
from sklearn.datasets import samples_generator, load_iris
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline

from tg.store import EstimatorStore, DisambiguatorStore


class TestEstimatorStore:
    
    def test_single_estimator(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        estimator = MultinomialNB()
        estimator.fit(X, y)  
        class_priors = estimator.class_log_prior_ 
        feat_prob = estimator.feature_log_prob_
        
        fname = tempfile.NamedTemporaryFile().name
        f = EstimatorStore(fname, "w")
        path = "nb"
        f.store_fit(estimator, path)
        f.close()
        
        estimator = MultinomialNB()
        f = EstimatorStore(fname)
        f.restore_fit(estimator, path)
        
        assert_array_equal(class_priors, estimator.class_log_prior_ )
        assert_array_equal(feat_prob, estimator.feature_log_prob_)
        
    def test_parameters(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        estimator = MultinomialNB(alpha=2.0, fit_prior=False)
        estimator.fit(X, y)  
        
        fname = tempfile.NamedTemporaryFile().name
        f = EstimatorStore(fname, "w")
        path = "nb"
        f.store_fit(estimator, path, set_params=True)
        f.close()
        
        estimator = MultinomialNB()
        f = EstimatorStore(fname)
        f.restore_fit(estimator, path, set_params=True)
        
        assert estimator.alpha == 2.0
        assert estimator.fit_prior is False
        
    def test_pipeline(self):
        X, y = samples_generator.make_classification(
            n_informative=5, n_redundant=0, random_state=42)

        anova_filter = SelectKBest(f_regression, k=5)
        clf = SVC(kernel='linear')
        anova_svc = Pipeline([('anova', anova_filter), 
                              ('svc', clf)])
        anova_svc.fit(X, y)
        score = anova_svc.score(X,y)
        
        fname = tempfile.NamedTemporaryFile().name
        f = EstimatorStore(fname, "w")
        path = "anova_svc"
        f.store_fit(anova_svc, path, set_params=True)
        f.close()
        
        # redefine
        anova_filter2 = SelectKBest(f_regression)
        clf2 = SVC()
        anova_svc2 = Pipeline([('anova', anova_filter2), 
                               ('svc', clf2)])
            
        f = EstimatorStore(fname)
        f.restore_fit(anova_svc2, path, set_params=True)
        
        score2 = anova_svc2.score(X,y)
        
        assert score == score2
        
        for attr in f.get_fitted_attrs(anova_filter):
            assert_array_equal( getattr(anova_filter, attr),
                                getattr(anova_filter2, attr) )
            
        for attr in f.get_fitted_attrs(clf):
            assert_array_equal( getattr(clf, attr),
                                getattr(clf2, attr) )
        
 
class TestDisambiguatorStore:
    
    def test_disambiguator_store(self):
        # Create a silly classifier that disambiguates between "stam" (tree
        # trunk) or "romp" (body trunk) as the Dutch translation of the
        # English noun "trunk"
        lempos = u"trunk/n"
        # FIXME: store_fit() should only accept unicode strings
        target_names = u"stam romp".encode("utf-8").split()
        vocab = u"boom hoofd".split()
        
        X = np.array([[0,1],
                      [1,0],
                      [0,1],
                      [1,0]])
        y = np.array([1,0,1,0])
        
        estimator = NearestCentroid()
        estimator.fit(X, y)
        
        centroids = estimator.centroids_
        score = estimator.score(X, y)
        
        # Store estimator
        fname = tempfile.NamedTemporaryFile().name
        f = DisambiguatorStore(fname, "w")
        f.save_estimator(NearestCentroid())
        f.save_vocab(vocab)
        f.store_fit(lempos, estimator)
        f.save_target_names(lempos, target_names)
        f.close()
        
        # Restore estimator    
        f2 = DisambiguatorStore(fname) 
        estimator2 = f2.load_estimator()
        vocab2 = f2.load_vocab()
        f2.restore_fit(lempos, estimator2)
        target_names2 = f2.load_target_names(lempos)
        centroids2 = estimator2.centroids_
        score2 = estimator2.score(X, y)
        
        assert_array_equal(centroids, centroids2)
        assert target_names == target_names2
        assert vocab == vocab2
        assert score == score2
       