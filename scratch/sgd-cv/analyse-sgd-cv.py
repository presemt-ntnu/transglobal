import numpy as np

r = np.load("_sgd-cv-1_results_de-en.npy")

# remove partial results
r = r[:-15]

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


    