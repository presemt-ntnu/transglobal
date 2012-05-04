#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
print samples
"""

import codecs
import sys

import h5py
from tg.utils import coo_matrix_from_hdf5


def print_samples(fname, lempos=None, outf=codecs.getwriter('utf8')(sys.stdout)):
    hdfile = h5py.File(fname, "r")
    vocab = [t.decode("utf-8") for t in hdfile["vocab"]]
    
    if lempos:
        lempos_list = [lempos]
    else:
        lempos_list  = [ lemma + u"/" + pos
                         for lemma in hdfile["samples"]
                         for pos in hdfile["samples"][lemma] ]
    
    for lempos in lempos_list:
        outf.write(78 * "=" + "\n"+ lempos + "\n" + 78 * "=" + "\n")
        group = hdfile["samples"][lempos]
        sample_mat = coo_matrix_from_hdf5(group)
        sample_mat = sample_mat.tocsr()
        for i, row in enumerate(sample_mat):
            outf.write(u"{0:<6d}: {1}\n".format(
                i, ", ".join(vocab[j] for j in row.indices)))
                
                
if __name__ == "__main__":
    import sys
    print_samples(*sys.argv[1:])
    
    
    