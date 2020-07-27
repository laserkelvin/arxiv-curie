from pathlib import Path
from tempfile import TemporaryDirectory

import arxiv

from curie.loaders import base
from curie.cleaners import base as clean_base


class ArxivReader(base.AbstractLoader):

    __name__ = "ArxivReader"

    def __init__(self, keep=False):
        super().__init__()
        self.keep = keep
        self.urls = list()
        self._results = list()

    @property
    def results(self):
        return self._results

    def query(
        self,
        query: str,
        max_results=50,
        sort_by="relevance",
        sort_order="descending",
        iterative=False,
        **kwargs
    ):
        self._results = arxiv.query(
            query,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
            iterative=iterative,
            **kwargs,
        )
        self.urls = [result.get("url", "") for result in self._results]
        self.urls = [clean_base.remove_arxiv_url(url) for url in self.urls]

    def retrieve(self):
        results = iter(self._results)
        if self.keep:
            save_dir = Path.cwd()
        else:
            save_dir = None
        with TemporaryDirectory(dir=save_dir) as temp_dir:
            for result in results:
                filename = arxiv.download(result, dirpath=temp_dir)
                yield self.read_pdf(filename)
