"""
approximate translation accuracy score
"""

import logging
import collections

from tg.mteval import read_ref_trans_counts


log = logging.getLogger(__name__)

Accuracy = collections.namedtuple("Accuracy",
                                  "correct incorrect ignored score")

def accuracy_score(graphs, ref_fname, score_attr):
    """
    Compute approximate accuracy score
    
    Parameters
    ----------
    graphs: list of TransGraph instances
    ref_fname: str
        name of file containing lemmatized reference translation 
        in mteval format
    score_attr: str
        scoring attribute on edge (normally a model score)
        
    Returns
    -------
    Accuracy(correct, incorrect, ignored, score): named tuple
        
    Notes
    -----
    The score is approximate, because there is no alignment between source
    and target lemmas, so we cannot be sure what the correct translation of a
    lemma is. However, if the predicted translation occurs anywhere in the
    reference translations of the sentence, we can guess that it is a correct
    translations. In practice, this works reasonably well for content words,
    which tend to occur just once in a sentence. Not do for function words
    like articles or pronouns, which are likely to occur multiple times in
    the same sentence. False positives are thus to be expected.
    
    If none of the translation edges for a source lemma contains the score
    attribute, it is assumed that there is no model/prediction for it,
    and it is ignored for the purpose of calculating accuracy.
    """
    ref_trans_counts = read_ref_trans_counts(ref_fname, flatten=True)
    correct, incorrect, ignored = 0, 0, 0
    
    for graph, lemma_counts in zip(graphs, ref_trans_counts):
        log.debug(graph)
        log.debug("source lemmas: {}".format(graph.source_lemmas()))
        log.debug("reference lemma counts: {}".format(lemma_counts))
        
        for u in graph.source_nodes_iter():
            log.debug(u"checking source node {!r} with lemma {!r}".format(
                u, graph.lemma(u)))
            score, v = graph.max_score(u, score_attr)
            if v:
                target_lemma = graph.lemma(v)
                log.debug("  best translation is node {!r} with lemma {!r} "
                          "({}={:.3f})".format(
                              v, target_lemma, score_attr, score))
                if target_lemma.lower() in lemma_counts:
                    correct += 1
                    log.debug("    which is correct :-)")
                else:
                    incorrect += 1
                    log.debug("    which is NOT correct :-(")
            else:
                ignored += 1
                log.debug("  none of its translation edges have score "
                          "attribute {!r}".format(score_attr))
                
                    
    score = correct / float(correct + incorrect)
    result = Accuracy(correct, incorrect, ignored, score)
    log.info(result)
    return result
                



    