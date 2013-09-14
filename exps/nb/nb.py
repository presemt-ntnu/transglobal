"""
Naive Bayes models
"""

import logging

from sklearn.feature_selection import SelectFpr, chi2
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

from tg.config import config
from tg.model import ModelBuilder, PriorModelBuilder
from tg.skl.selection import MinCountFilter, MaxFreqFilter
from tg.exps.support import grid_search


log = logging.getLogger(__name__)


def nb_build_models(ns):
    """
    build either normal NB models or with class priors from corpus counts
    """
    ns.models_fname = ns.fname_prefix + "_models.hdf5"           
    
    if getattr(ns, "class_priors"):
        counts_fname = config["count"]["lemma"][ns.target_lang]["pkl_fname"]       
        model_builder = PriorModelBuilder(ns.ambig_map, 
                                          ns.samples_fname,
                                          ns.models_fname, 
                                          ns.classifier, 
                                          counts_fname=counts_fname)
    else:
        model_builder = ModelBuilder(ns.ambig_map, 
                                     ns.samples_fname, 
                                     ns.models_fname,
                                     ns.classifier)
    model_builder.run()
    # clean up params
    delattr(ns, "ambig_map")
    

@grid_search    
def nb_classifier(classifier=MultinomialNB,
                  min_count=5,
                  max_freq=0.1,
                  chi2_alpha=0.05,
                  nb_alpha=1.0):
    """
    construct pipeline with feature selection and NB classifier
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
