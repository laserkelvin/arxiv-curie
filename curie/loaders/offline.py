from pathlib import Path

from curie.loaders import base


class OfflineReader(base.AbstractLoader):
    def __init__(self):
        super().__init__()
        self._results = list()

    @property
    def results(self):
        return self._results

    def query(self, path: str):
        paths = Path(path).rglob("*.pdf")
        self.results = paths
        return paths

    def retrieve(self, paths=None):
        if paths is None:
            paths = self.results
        for path in paths:
            if path not in self.results:
                self.results.append(str(path))
            yield self.read_pdf(path)
