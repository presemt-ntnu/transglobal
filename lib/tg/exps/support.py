"""
support for experiments
"""

import cPickle
import logging

import numpy as np

import asciitable as at

log = logging.getLogger(__name__)


class ResultsStore(object):
    
    default_dtype = "f"
    
    def __init__(self, descriptor, fname_prefix):
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
                
        self.results = np.zeros(9999, dtype=dtype)
        self.npz_fname = fname_prefix + ".npz"
        self.txt_fname = fname_prefix + ".txt"
        self.count = 0
            
    def append(self, ns):
        # get field values by evaluating getter expression
        env = {"ns": ns}
        self.results[self.count] = tuple(eval("ns." + expr, env) 
                                         for expr in self.getters)
        self.count += 1
        # save each intermediary result
        log.info("saving numpy results to " + self.npz_fname)
        np.save(self.npz_fname, 
                self.results[:self.count]) 
        log.info("saving text results to " + self.txt_fname)
        at.write(self.results[:self.count], 
                 self.txt_fname, 
                 Writer=at.FixedWidthTwoLine, 
                 delimiter_pad=" ")


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
                


        
