"""
test ResultsStore
"""

from tempfile import NamedTemporaryFile

from tg.exps.support import ResultsStore, Namespace


class TestResultStore:
    
    def test_grow(self):
        descriptor = [("a", "f"), ("b", "S8")]
        fname_prefix = NamedTemporaryFile().name
        buf_size = 5
        store = ResultsStore(descriptor, fname_prefix, buf_size=5)
        ns = Namespace(a=1, b="x")
        for _ in range(buf_size):
            store.append(ns)
        assert store.results.shape[0] == buf_size
        # results is filled - appening one more must grow results
        store.append(ns)
        assert store.results.shape[0] == 2 * buf_size
