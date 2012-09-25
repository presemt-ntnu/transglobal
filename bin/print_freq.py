#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Print frequency (counts) of lemmas in samples to stdout in utf-8 char encoding
"""

import argparse
import codecs
import sys

import h5py
from tg.utils import coo_matrix_from_hdf5


def print_freq(samp_fname, lemma=None, pos=None, minimum=0,
               outf=codecs.getwriter('utf8')(sys.stdout)):
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
                # dtype must be int rather than i8, because calling .sum will return 
                sample_mat = coo_matrix_from_hdf5(group, dtype="f")
                sample_mat = sample_mat.tocsr()
                sums = sample_mat.sum(axis=0)
                total = float(sums.sum())
                indices = sums.argsort()
                indices = indices[0,:].tolist()[0]
                indices.reverse()
                
                for i in indices:
                    if sums[0,i] > minimum:
                        outf.write(u"{0:>16.8}{1:>16.8f}%     {2}\n".format(
                            sums[0,i],
                            (100 * sums[0,i]) / total,
                            vocab[i]))
    

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

parser.add_argument(
    "-m", "--minimum",
    metavar="N",
    type=int,
    default=0,
    help="minimum frequency of lemma")

args = parser.parse_args()

print_freq(args.samp_fname, args.lemma, args.pos, args.minimum)