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
    
    
def coo_matrix_from_hdf5(group, dtype=None):
    """
    read a sparse matric in COO format from a HDF5 group
    """
    return sp.coo_matrix((group["data"], group["ij"]), 
                         shape=group.attrs["shape"], dtype=dtype)



class Namespace(object):
    """
    Simple namespace for passing around variables and functions
    """
    
    def __init__(self, **kwargs):
        # copy keyword args to object's dict so they becomes attributes
        self.__dict__.update(**kwargs)
        
    def __repr__(self):
        return (
            self.__class__.__name__ +
            "(" +
            ", ".join("{}={}".format(k,v) 
                      for k, v in self.__dict__.items()
                      if not k.startswith("__")) + 
            ")" )
    
    def __str__(self):
        return (
            self.__class__.__name__ +
            "(\n" +
            ",\n".join("  {:<16} = {}".format(k,v) 
                      for k, v in sorted(self.__dict__.items())
                      if not k.startswith("__")) + 
            "\n)" )
        
    def import_module(self, mod_name, condition=callable):
        """
        Import names from a module
        
        Params
        ------
        mod_name: str
            module name (not module object)
        condition: function or None, optional
            test for importing objects;
            function is called with object and should return boolean;
            default is to import functions only
        """
        module = __import__(mod_name)
        # drill down to subpackage
        for name in mod_name.split(".")[1:]:
            module = getattr(module, name)
        
        for name in dir(module):
            obj = getattr(module, name)
            if condition is None or condition(obj):
                setattr(self, name, obj)
                
    def import_locals(self, local_ns, kwargs="kwargs"):
        """
        Import names from local environment (=locals())
        """
        for name, obj in local_ns.items():
            # prevent recursive import if this namespace is in local namespace
            if not obj is self:
                if name == kwargs:
                    self.__dict__.update(**obj)
                else:
                    setattr(self, name, obj)
                


def grid_search(process, *args, **kwargs):
    """
    Explore parameter space
    """
    # check for grid parameters starting with an underscore
    for keyword in kwargs:
        if keyword.startswith("_"):
            # obtain real parameter by stripping prefix
            param = keyword.lstrip("_")
            # remove grid param
            for value in kwargs.pop(keyword):
                # insert real param
                kwargs[param] = value 
                # recursively handle other grid params (if any)
                for exp in grid_search(process, *args, **kwargs):
                    yield exp
            break
    else:
        # no more grid params in kwargs; 
        # terminate recursion with single experiment
        yield process(*args, **kwargs)        
        
