import logging
import codecs
import sys

from os import makedirs
from os.path import exists, dirname

import numpy as np
import scipy.sparse as sp


def create_dirs(path):
    """
    create directores if they do not exists yet;
    path is assumed to be a path to a *file* (not a dir)
    """
    path = dirname(path)        
    if not exists(path):
        makedirs(path)
        
        
def set_default_log(level=logging.INFO, encoding="utf-8", errors="strict"):
    """
    configure default "root" logger
    """ 
    # get root logger
    log = logging.getLogger()
    # output to wrapped stderr
    stream = codecs.getwriter(encoding)(sys.stderr, errors=errors)
    log_handler = logging.StreamHandler(stream)
    log_format = logging.Formatter('| %(levelname)-8s | %(name)-24s | %(message)s')
    log_handler.setFormatter(log_format)
    log.addHandler(log_handler)
    log.setLevel(level)
    return log


def indent(elem, level=0):
    """
    elementree formatting: indent element and all subelements for pretty
    printing of XML
    """
    # copied from Fredrik Lund 
    # http://effbot.python-hosting.com/file/effbotlib/ElementTree.py
    i = "\n" + level*"  "
    
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            indent(e, level+1)
        if not e.tail or not e.tail.strip():
            e.tail = i

    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i



# Sparse matrices

def coo_matrix_to_hdf5(matrix, group, data_dtype=None, **kwargs):
    """
    store a sparse matrix in COO format in a HDF5 group

    dtype specifies the type for data values
    
    Additional kwargs (e.g. compression="gzip") are passed to the dataset
    contructors for both the data values and the indices
    """
    # Non-zero entries of the sparse matrix are contained in three
    # equally-sized arrays which specify row indices, column indices and data
    # values. The first two are integers and can thus be combined into a
    # single dense matrix. The third one can contains items of different type
    # and must thus be stored seperately. The shape of the sparse matrix must
    # be preserved as well.
    group.create_dataset("data", data=matrix.data, dtype=data_dtype, **kwargs)
    index_data = np.array([matrix.row, matrix.col])
    group.create_dataset("ij", data=index_data, **kwargs)
    group.attrs["shape"] = matrix.shape
    
    
def coo_matrix_from_hdf5(group):
    """
    read a sparse matric in COO format from a HDF5 group
    """
    return sp.coo_matrix((group["data"], group["ij"]), 
                         shape=group.attrs["shape"])
