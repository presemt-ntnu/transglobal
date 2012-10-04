"""
graph process
"""

import transgraph


class GraphProcess(object):
    """
    abstract base class for TransGraph processors
    """
    
    def __init__(self,  *args, **kwargs):
        pass
                 
    
    def __call__(self, obj, *args, **kwargs):
        """
        run process with either a single obj or a list of objects 
        """
        if isinstance(obj, basestring) or isinstance(obj, transgraph.TransGraph):
            return self._single_run(obj, *args, **kwargs)
        else:
            return self._batch_run(obj, *args, **kwargs)


    def _single_run(self, obj, *args, **kwargs):
        return NotImplemented
    
    def _batch_run(self, obj_list, *args, **kwargs):
        return [ self._single_run(obj, *args, **kwargs)
                 for obj in obj_list ]
            