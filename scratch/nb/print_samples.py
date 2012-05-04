#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
print samples
"""

import h5py
from tg.utils import coo_matrix_from_hdf5


def print_samples(fname, lempos=None):
    hdfile = h5py.File(fname, "r")
    vocab = [t.decode("utf-8") for t in hdfile["vocab"]]
    
    if lempos:
        lempos_list = [lempos]
    else:
        lempos_list  = [ lemma + u"/" + pos
                         for lemma in hdfile["samples"]
                         for pos in hdfile["samples"][lemma] ]
    
    for lempos in lempos_list:
        print 78 * "="
        print lempos
        print 78 * "="
        group = hdfile["samples"][lempos]
        sample_mat = coo_matrix_from_hdf5(group)
        sample_mat = sample_mat.tocsr()
        for i, row in enumerate(sample_mat):
            print "{0:<6d}: {1}".format(
                i, ", ".join(vocab[j] for j in row.indices))
                
if __name__ == "__main__":
    import sys
    print_samples(*sys.argv[1:])
    
    
    