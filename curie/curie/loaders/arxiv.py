
from tempfile import TemporaryDirectory

import arxiv
from tika import parser

from .base import AbstractLoader


class ArxivReader(AbstractLoader):
    def __init__(self, keep=False):
        super().__init__()
        self.keep = keep

    def query(
        self,
        query: str,
        max_results=50,
        sort_by="relevance",
        sort_order="descending",
        iterative=True,
        **kwargs
    ):
        path = TemporaryDirectory()
        self.results = arxiv.query(
            query,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
            iterative=iterative ** kwargs,
        )
    
    def retrieve(self):
        results = iter(self.results)
        while True:
            filename = arxiv.download(result, dirpath=path.name)
            paper = parser.from_file(filename)
            yield paper["content"]
            

