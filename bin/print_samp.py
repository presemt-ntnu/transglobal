#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Print samples to stdout in utf-8 char encoding
"""

import argparse
import codecs
import sys

import h5py
from tg.utils import coo_matrix_from_hdf5


def print_samples(samp_fname, lemma=None, pos=None, outf=codecs.getwriter('utf8')(sys.stdout)):
    hdfile = h5py.File(samp_fname, "r")
    vocab = [t.decode("utf-8") for t in hdfile["vocab"][()]]
    line = 78 * "=" + "\n"
    
    if lemma:
        if lemma in hdfile["samples"]:
            lemma_list = [lemma]
        else:
            lemma_list = []
    else:
        lemma_list = hdfile["samples"]
    
    for lemma in lemma_list:
        for lemma_pos in hdfile["samples"][lemma]:
            if not pos or lemma_pos == pos:
                outf.write(line + lemma + u"/" + lemma_pos + "\n" + line)
                group = hdfile["samples"][lemma][lemma_pos]
                sample_mat = coo_matrix_from_hdf5(group)
                sample_mat = sample_mat.tocsr()
                for count, row in enumerate(sample_mat):
                    outf.write(u"{0:<9d}: ".format(count + 1))
                    pairs = [ (vocab[j], count) 
                              for j, count in zip(row.indices, row.data) ]
                    pairs.sort()
                    string = u", ".join(u"{}:{}".format(*p) for p in pairs)
                    outf.write(string + u"\n")
    

parser = argparse.ArgumentParser(description=__doc__)     
    
parser.add_argument(
    "samp_fname",
    metavar="SAMPLES_FILE",
    help="samples file HDF5 format")

parser.add_argument(
    "-l", "--lemma",
    metavar="LEMMA",
    help="lemma (unknown lemma results in no output)")

parser.add_argument(
    "-p", "--pos",
    metavar="POS",
    help="part-of-speech tag (unknown tag results in no outut)")

args = parser.parse_args()

# work around because it seems argparse does not convert strings to unicode
if isinstance(args.lemma, basestring):
    args.lemma = args.lemma.decode(sys.stdin.encoding)
    
if isinstance(args.pos, basestring):
    args.pos = args.pos.decode(sys.stdin.encoding)

print_samples(args.samp_fname, args.lemma, args.pos)