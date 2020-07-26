from pathlib import Path

from curie.curie.loaders import base


class OfflineReader(base.AbstractLoader):
    def __init__(self):
        super().__init__()
        self.results = list()

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
