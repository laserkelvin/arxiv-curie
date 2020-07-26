from pathlib import Path
from tempfile import TemporaryDirectory

import arxiv

from curie.curie.loaders import base
from curie.curie.cleaners import base as clean_base


class ArxivReader(base.AbstractLoader):
    def __init__(self, keep=False):
        super().__init__()
        self.keep = keep
        self.urls = list()
        self.results = list()

    def query(
        self,
        query: str,
        max_results=50,
        sort_by="relevance",
        sort_order="descending",
        iterative=False,
        **kwargs
    ):
        self.results = arxiv.query(
            query,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
            iterative=iterative,
            **kwargs,
        )
        self.urls = [result.get("url") for result in self.results]
        self.urls = [clean_base.remove_arxiv_url(url) for url in self.urls]

    def retrieve(self):
        results = iter(self.results)
        if self.keep:
            save_dir = Path.cwd()
        else:
            save_dir = None
        with TemporaryDirectory(dir=save_dir) as temp_dir:
            for result in results:
                filename = arxiv.download(result, dirpath=temp_dir)
                yield self.read_pdf(filename)
