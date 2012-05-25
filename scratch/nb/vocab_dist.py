"""

"""

import cPickle

import matplotlib.pyplot as plt
import numpy as np

import h5py

from tg.config import config


filtered_hdf_fname = "en_samples_filtered.hdf5"
f = h5py.File(filtered_hdf_fname)

counts_pkl_fname = config["count"]["lemma"]["en"]["pkl_fname"]
counts_dict = cPickle.load(open(counts_pkl_fname))


vocab_counts = [ (counts_dict[lemma.decode("utf-8")], lemma.decode("utf-8")) 
                 for lemma in f["vocab"] ]
total = float(sum(zip(*vocab_counts)[0]))
vocab_counts.sort(reverse=True)
cum_sum = 0

for i, (count, lemma) in enumerate(vocab_counts):
    cum_sum += count
    print "{0:<12d}{1:<24s}{2:>12d}{3:12.4f}{4:12.4f}".format(
        i +1,
        lemma, 
        count,
        (count/total) * 100,
        (cum_sum/total) * 100
        )
print        

#plt.hist(counts, bins=100, range=(100,10000))
#plt.hist(counts, bins=100, range=(11,100))
#plt.hist(counts, bins=100, range=(1000, counts.max()), cumulative=True)
#plt.show()

