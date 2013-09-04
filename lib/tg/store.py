# -*- coding: utf-8 -*-

"""
storage of estimators and disambiguators in hdf5 file
"""

# TODO:
# - add doc strings and comments
# - need to import estimator classes when unpickling? See importlib.import_module 
# - close hdf5 file at exit?
# - how to set dataset options per fit attrib, e.g. chunk shape
# - are there any estimators for which we need to store sparse matrices?



import cPickle
import h5py
import logging

import numpy as np

from sklearn.pipeline import Pipeline

log = logging.getLogger(__name__)



# create new data type for variable-length strings in HDF5
# see http://code.google.com/p/h5py/wiki/HowTo#Variable-length_strings
VARSTR = h5py.new_vlen(str)


class EstimatorStore(object):
    
    # key in group's attrs for storing estimator's parameters
    PARAMS_KEY = "params"
    
    # affix atached to name of subestimators in a pipeline
    SUB_AFFIX = "__"
    
    # attributes set by an estimator's fit() method,
    # i.e. those which need to be (re)stored
    FITTED_ATTRS = {
        "BernoulliNB": ("class_log_prior_", "feature_log_prob_",),
        "CosNearestCentroid": ("centroids_", "classes_"),
        "MinCountFilter": ("feature_importances_",),
        "DecisionTreeClassifier": ("tree_"),
        "MaxFreqFilter": ("feature_importances_",),
        "MultinomialNB": ("class_log_prior_", "feature_log_prob_",), 
        "NearestCentroid": ("centroids_", "classes_"),
        "NearestCentroidProb": ("centroids_", "classes_"),
        "Normalizer": (),
        "NMF": ("components_", "n_components_"),
        "Pipeline": (),
        "SelectKBest": ("pvalues_", "scores_"),
        "SelectFpr": ("pvalues_", "scores_"),
        "SVC": ("shape_fit_", "support_", "support_vectors_", "n_support_", 
                "dual_coef_", "_intercept_", "classes_", "probA_", "probB_",
                "_label", "_gamma", "_sparse"),
        "TfidfTransformer": ("idf_",),
        }
    
    def __init__(self, file, mode=None, compression="lzf"):
        if not isinstance(file, h5py.File):  
            log.info("opening store file " + file)
            self.file = h5py.File(file, mode)
        else:
            self.file = file

        self.compression = compression
        
    def close(self):
        log.info("closing store file " + self.file.filename)
        self.file.close()
    
    def store_fit(self, estimator, path, set_params=False):
        self._store_estimator_fit(estimator, path, set_params)
        
        if isinstance(estimator, Pipeline):
            self._store_pipeline_fit(estimator, path)
            
    def _store_estimator_fit(self, estimator, path, set_params=False):
        # path must not exists yet, otherwise a valueError is raised
        self.file.create_group(path)
        
        if set_params:
            params = estimator.get_params()
            params_pkl = cPickle.dumps(params)
            self.file[path].attrs[self.PARAMS_KEY] = params_pkl
            
        for attr in self.get_fitted_attrs(estimator):
            data = getattr(estimator, attr)
            try:
                dataset = self.file[path].create_dataset(attr, 
                                                    data=data,
                                                    compression=self.compression)
            except TypeError:
                # TypeError: Scalar datasets don't support chunk/filter options
                # from _hl/filters.py", line 71, in generate_dcpl
                # Try again without compression
                dataset = self.file[path].create_dataset(attr, data=data)
            
    def _store_pipeline_fit(self, estimator, path):
        for name, sub_estimator in estimator.steps:
            sub_path = path + "/" + name + self.SUB_AFFIX
            self._store_estimator_fit(sub_estimator, sub_path)
    
    def restore_fit(self, estimator, path, set_params=False):
        self._restore_estimator_fit(estimator, path, set_params)
        
        if isinstance(estimator, Pipeline):
            self._restore_pipeline_fit(estimator, path)
            
    def _restore_estimator_fit(self, estimator, path, set_params=False):
        if set_params:
            params_pkl = self.file[path].attrs[self.PARAMS_KEY]
            params = cPickle.loads(params_pkl)
            estimator.set_params(**params)
        
        for attr in self.get_fitted_attrs(estimator):
            dataset = self.file[path][attr]
            if dataset.shape == ():
                # value us scalar 
                value = dataset.value
            elif len(dataset):                
                # value is array                
                value = dataset[:]
            else:
                # Workaround for problem that conversion to an numpy array does
                # not work when dataset with shape like (0,). This occurs e.g.
                # with SVC.probA_ and SVC.probB                
                value = np.zeros(shape=dataset.shape, dtype=dataset.dtype)
            
            setattr(estimator, attr, value)
            
    def _restore_pipeline_fit(self, estimator, path):
        for name, sub_estimator in estimator.steps:
            sub_path = path + "/" + name + self.SUB_AFFIX
            self._restore_estimator_fit(sub_estimator, sub_path)
            
    def get_fitted_attrs(self, estimator):
        class_name = estimator.__class__.__name__
        # if estimator is unknown, raise exception
        return self.FITTED_ATTRS[class_name]
    
    
    
class DisambiguatorStore(EstimatorStore):
    """
    Class for storing word translation disambiguation models per lempos. 
    This includes estimator instance, vocabulary and target names.
    """
    
    ESTIMATOR_PKL_ATTR = "estimator_pkl"
    VOCAB_PATH = "vocab"
    TARGET_NAMES_ATTR = "target_names"
    FITS_PATH = "fits"
    VOCAB_MASK_PATH = "vocab_mask"
        
    def save_estimator(self, estimator):
        # Pickle classifier and include in hdf5 file. This saves the
        # parameters from __init__. This should be dome *before* a call to
        # fit(), so any attributes changed by calling fit() are excluded.
        # Loading this pickled object requires its class to be part of the
        # current namespace. Alternative is to use the _get_params() and
        # set_params() methods from the BaseEstimator class
        log.info("saving pickled estimator {0}".format(estimator))
        estimator_pkl = cPickle.dumps(estimator)
        self.file.attrs[self.ESTIMATOR_PKL_ATTR] = estimator_pkl
        
    def load_estimator(self):
        estimator_pkl = self.file.attrs[self.ESTIMATOR_PKL_ATTR]
        estimator =  cPickle.loads(estimator_pkl)                
        log.info("loaded pickled estimator {0}".format(estimator))
        return estimator
      
    def save_vocab(self, vocab):
        log.info("saving vocabulary ({0} terms)".format(len(vocab)))
        vocab = [ lemma.encode("utf-8") for lemma in vocab ]
        self.file.create_dataset(self.VOCAB_PATH,
                                 data=vocab,
                                 dtype=VARSTR,
                                 compression=self.compression)     
        
    def load_vocab(self, as_dict=False):
        utf8_vocab = self.file[self.VOCAB_PATH][()]
        
        if as_dict:
            vocab = dict((lemma.decode("utf-8"), i) 
                         for i,lemma in enumerate(utf8_vocab))
        else:
            vocab = [ lemma.decode("utf-8") 
                      for lemma in utf8_vocab ]
            
        log.info("loaded vocabulary ({0} terms)".format(len(vocab)))
        return vocab
    
    def copy_vocab(self, sample_hdf_file):
        """
        Copy vocabulary from sample file to store file
        
        Parameters
        ----------
        sample_hdf_file: h5py.File instance
            hdf5 file containing samples and vocabulary under path VOCAB_PATH
        """
        log.info("copying vocabulary from sample file")
        sample_hdf_file.copy(self.VOCAB_PATH, self.file)
        
    def save_vocab_mask(self, lempos, mask):
        log.debug(u"saving vocab mask for lempos {0}".format(lempos))
        path = self.FITS_PATH + "/" + lempos + "/" + self.VOCAB_MASK_PATH
        self.file[path] = mask
        
    def load_vocab_mask(self, lempos):
        log.debug(u"loading vocab mask for lempos {0}".format(lempos))
        path = self.FITS_PATH + "/" + lempos + "/" + self.VOCAB_MASK_PATH
        return self.file[path]
    
    def save_target_names(self, lempos, target_names):
        log.debug(u"saving target names for lempos {0}".format(lempos))
        path = self.FITS_PATH + "/" + lempos
        self.file[path].attrs[self.TARGET_NAMES_ATTR] = target_names
        
    def load_target_names(self, lempos):
        log.debug(u"loading target names for lempos {0}".format(lempos))
        path = self.FITS_PATH + "/" + lempos
        target_names  = self.file[path].attrs[self.TARGET_NAMES_ATTR] 
        return [ name.decode("utf-8") for name in target_names ]
    
    def store_fit(self, lempos, estimator, set_params=False):
        path = self.FITS_PATH + "/" + lempos
        EstimatorStore.store_fit(self, estimator, path, set_params)
    
    def restore_fit(self, lempos, estimator, set_params=False):
        path = self.FITS_PATH + "/" + lempos
        EstimatorStore.restore_fit(self, estimator, path, set_params)

