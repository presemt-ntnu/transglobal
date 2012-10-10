
import codecs
import sys

import numpy as np

from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import normalize
from sklearn.utils.validation import atleast2d_or_csr

from tg.store import DisambiguatorStore


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
        


# Tools

def print_centroids(models_fname, lemma=None, pos=None, minimum=0, n=None,
                    outf=codecs.getwriter('utf8')(sys.stdout) ):
    # If used in combination with a feature selector,
    # models must be build using with_vocab_mask=True
    
    # FIXME: messy code below
    models = DisambiguatorStore(models_fname)
    classifier = models.load_estimator()
    full_vocab = np.array(models.load_vocab())
    fits = models.file[models.FITS_PATH]
    line = 78 * "=" + "\n"
    subline = "    " + 74 * "-" + "\n"
    if isinstance(outf, basestring):
        outf = codecs.open(outf, "w", encoding="utf-8")
        
    if lemma:
        if lemma in fits:
            lemma_list = [lemma]
        else:
            lemma_list = []
    else:
        lemma_list = fits
    
    for lemma in lemma_list:
        for lemma_pos in fits[lemma]:
            if not pos or lemma_pos == pos:
                lempos = lemma + u"/" + lemma_pos
                outf.write(line + lempos + "\n" + line)
                models.restore_fit(lempos, classifier)
                
                if isinstance(classifier, NearestCentroid):
                    centroids_ = classifier.centroids_
                else:
                    # NearestCentroid is last item in Pipeline
                    nc = classifier.steps[-1][-1]
                    assert isinstance(nc, NearestCentroid)
                    centroids_ = nc.centroids_
                    
                target_names = models.load_target_names(lempos)
                target_n = 0
                
                try:
                    vocab_mask = models.load_vocab_mask(lempos)[:]
                except:
                    vocab = full_vocab
                else:
                    vocab = full_vocab[vocab_mask]
                    
                
                for target, centroid in zip(target_names, centroids_):
                    target_n += 1
                    outf.write(subline)
                    outf.write(u"    [{}] {} ---> {}\n".format(
                        target_n,
                        lempos,
                        target))
                    outf.write(subline)
                    indices = centroid.argsort().tolist()
                    indices.reverse()
                    
                    for i in indices[:n]:
                        if centroid[i] > minimum:
                            outf.write(u"    {0:>16.8f}    {1}\n".format(
                                centroid[i],
                                vocab[i]))