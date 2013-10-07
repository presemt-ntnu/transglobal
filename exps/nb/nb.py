"""
Naive Bayes models
"""

import logging

from sklearn.feature_selection import SelectFpr, chi2
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

from tg.config import config
from tg.skl.selection import MinCountFilter, MaxFreqFilter
from tg.classcore import Vectorizer
from tg.exps.support import grid_search


log = logging.getLogger(__name__)


# grid_search returns an *iterator* over classifiers
@grid_search    
def nb_classifier(classifier=MultinomialNB,
                  min_count=5,
                  max_freq=0.1,
                  chi2_alpha=0.05,
                  nb_alpha=1.0):
    """
    Construct pipeline with feature selection and NB classifier
    """
    components = []
    
    if min_count:
        components.append(("MCF", MinCountFilter(min_count)))
    if max_freq:
        components.append(("MFF", MaxFreqFilter(max_freq)))
    if chi2_alpha:
        components.append(("CHI2", SelectFpr(chi2, alpha=chi2_alpha)))
    
    components.append(("MNB", classifier(alpha=nb_alpha)))
    return Pipeline(components)


@grid_search
def vectorizer(score_attr=None, min_score=None):
    return Vectorizer(score_attr=score_attr, min_score=min_score)
    
