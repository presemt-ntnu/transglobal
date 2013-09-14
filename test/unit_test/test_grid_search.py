"""
test grid_search function
"""

from tg.exps.support import grid_search_func, grid_search


def exp(*args, **kwargs):
    return locals() 

@grid_search
def exp2(*args, **kwargs):
    return locals() 



def test_grid_search_func():    
    exps = grid_search_func(exp,
                       "p1",
                       "_p2",
                       _k1=(1,2),
                       k2=(3,4),
                       _k3=("a","b"))
    result = list(exps)
    expected = [
        {'args': ('p1', '_p2'), 'kwargs': {'k1': 1, 'k2': (3, 4), 'k3': 'a'}},
        {'args': ('p1', '_p2'), 'kwargs': {'k1': 1, 'k2': (3, 4), 'k3': 'b'}},
        {'args': ('p1', '_p2'), 'kwargs': {'k1': 2, 'k2': (3, 4), 'k3': 'a'}},
        {'args': ('p1', '_p2'), 'kwargs': {'k1': 2, 'k2': (3, 4), 'k3': 'b'}}]
    assert result == expected
    
    
def test_grid_search_decorator():
    
    exps = exp2("p1", 
                "_p2", 
                _k1=(1,2), 
                k2=(3,4), 
                _k3=("a","b"))
    result = list(exps)
    expected = [
        {'args': ('p1', '_p2'), 'kwargs': {'k1': 1, 'k2': (3, 4), 'k3': 'a'}},
        {'args': ('p1', '_p2'), 'kwargs': {'k1': 1, 'k2': (3, 4), 'k3': 'b'}},
        {'args': ('p1', '_p2'), 'kwargs': {'k1': 2, 'k2': (3, 4), 'k3': 'a'}},
        {'args': ('p1', '_p2'), 'kwargs': {'k1': 2, 'k2': (3, 4), 'k3': 'b'}}]
    assert result == expected    


