
import codecs
import sys

import numpy as np

from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import normalize
from sklearn.utils.validation import atleast2d_or_csr
from sklearn.metrics.pairwise import pairwise_distances

from tg.store import DisambiguatorStore


class NearestCentroidProb(NearestCentroid):
    """
    Variant of nearest centroid classifier that provides a predict_proba()
    method
    """
    
    def predict_proba(self, X):
        X = atleast2d_or_csr(X)
        if not hasattr(self, "centroids_"):
            raise AttributeError("Model has not been trained yet.")
        distances = pairwise_distances(X, self.centroids_, metric=self.metric)
        # This is evidently silly, as normalizing the distances does not turn
        # them into probabilities. However, we need this method in
        # tg.classify.TranslationClassifier._predict()
        normalize(distances, norm="l1", copy=False)
        # turn into probability of *similarity*
        probs = 1.0 - distances
        return probs
    

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
                            outf.write(u"    {0:>16.8f}    {1:<16}    {2}\n".format(
                                centroid[i],
                                vocab[i],
                                centroid[i] * 100 * "*"))