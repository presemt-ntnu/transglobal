import numpy as np

from tg.utils import text_table


r = np.load("_sgd-cv-1_results_de-en.npy")

# remove partial results
r = r[:-15]


def summary():
    params = ["alpha", "loss", "n_iter", "penalty"]
    scores = ["prec", "rec", "f-score", "accuracy"]
    
    
    
    keys = np.unique(r[params])
    
    summary = np.zeros(len(keys), r.dtype.descr[3:])
    
    for i, k in enumerate(keys):
        subset = r[r[params] == k]
        subset_scores = subset[scores]
        view = subset_scores.view(("f", len(subset_scores.dtype.names)))
        means = view.mean(axis=0)
        summary[i] = tuple(k) + tuple(means)
    
    
    summary.sort(axis=0, order=["f-score"])
    text_table(summary[::-1])
    print


def max_scores():
    subset = r[(r["alpha"] == 0.001) &
               (r["loss"] == "log") &
               (r["n_iter"] == 5) &
               (r["penalty"] == "l2")]
    subset.sort(axis=0, order=["f-score"])
    text_table(subset[::-1])
    print
    

if __name__ == "__main__":
    summary()
    max_scores()
    