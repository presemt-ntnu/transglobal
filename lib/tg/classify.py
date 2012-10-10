"""
classification of translation candidates
"""


import logging

from tg.store import DisambiguatorStore

    
log = logging.getLogger(__name__)


class TranslationClassifier(object):
    """
    Class for translation disambiguation using per lempos disambiguators
    stored in a HDF5 file
    """
    
    def __init__(self, models_fname):
        self.models = DisambiguatorStore(models_fname)
        self.classifier = self.models.load_estimator()
        self.vocab = self.models.load_vocab(as_dict=True)
        
    def score(self, source_lempos, context_vec):
        """
        score translation candidates for source lempos combination,
        returning a dict mapping target lempos combinations to scores
        """
        try:
            self.models.restore_fit(source_lempos, self.classifier)
        except KeyError:
            log.debug(u"no model available for source lempos " + 
                      source_lempos)
            return {}  
        
        target_names = self.models.load_target_names(source_lempos)
        return self._predict(context_vec, target_names)
    
    def _predict(self, context_vec, target_names):
        """
        return a dict mapping target names to scores
        """
        # FIXME: some estimators have no predict_proba method
        # e.g. NearestCentroid
        preds = self.classifier.predict_proba(context_vec)
        return dict(zip(target_names, preds[0]))
    