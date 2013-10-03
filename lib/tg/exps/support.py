"""
support for experiments
"""

import codecs
import cPickle
import logging
import os
import shutil

import numpy as np

from tg.utils import text_table

log = logging.getLogger(__name__)


def remove_exp_dir(name):
    exp_dir = "_" + name
    if os.path.exists(exp_dir):
        log.info("removing exp dir " + exp_dir)
        shutil.rmtree(exp_dir)  


class ResultsStore(object):
    
    default_dtype = "f"
    
    def __init__(self, descriptor, fname_prefix, buf_size=1000):
        self.getters = []
        dtype = []
        
        for elem in descriptor:
            if len(elem) == 1:
                dtype.append((elem[0] , self.default_dtype))
                self.getters.append(elem[0])
            elif len(elem) == 2:
                dtype.append(elem)
                self.getters.append(elem[0])
            else:
                dtype.append(elem[:2])
                self.getters.append(elem[2])
                
        self.results = np.zeros(buf_size, dtype=dtype)
        self.npy_fname = fname_prefix + ".npy"
        self.txt_fname = fname_prefix + ".txt"
        self.buf_size = buf_size
        self.count = 0
            
    def append(self, ns):
        # grow results array?
        if self.count == self.results.shape[0]:
            new_size = self.results.shape[0] + self.buf_size,
            self.results.resize(new_size, refcheck=False)
        # get field values by evaluating getter expression
        env = {"ns": ns}
        for i, expr in enumerate(self.getters):
            try:
                val = eval("ns." + expr, env) 
            except Exception, exception:
                log.warning("evaluating descriptor expression {!r} "
                            "raises exception {!r}".format(expr, exception))
                val = None
            self.results[self.count][i] = val 
        self.count += 1
        # save each intermediary result
        log.info("saving numpy results to " + self.npy_fname)
        np.save(self.npy_fname, 
                self.results[:self.count]) 
        log.info("saving text results to " + self.txt_fname)
        text_table(self.results[:self.count], self.txt_fname)


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


def grid_search_func(process, *args, **kwargs):
    """
    Explore parameter space
    """
    # check for grid parameters starting with an underscore
    for keyword in kwargs:
        if keyword.startswith("_"):
            log.debug("expanding keyword " + keyword)
            # obtain real parameter by stripping prefix
            param = keyword.lstrip("_")
            # remove grid param
            for value in kwargs.pop(keyword):
                # insert real param
                kwargs[param] = value 
                log.debug("{} = {}".format(param, value))
                # recursively handle other grid params (if any)
                for output in grid_search_func(process, *args, **kwargs):
                    yield output
            break
    else:
        # no more grid params in kwargs; 
        # terminate recursion with single experiment
        log.debug("terminating with args={} and kwargs={}".format(args, kwargs))
        yield process(*args, **kwargs)

        
                

def grid_search(func):
    """
    decorator for grid_search_func
    """
    def wrapper(*args, **kwargs):
        return grid_search_func(func, *args, **kwargs)
    return wrapper
        
