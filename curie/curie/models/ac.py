
from typing import Union, List
from pathlib import Path, PosixPath

from curie.curie.models import base
from curie.curie.loaders import arxiv, offline
from curie.curie.cleaners import nltk
from curie.curie.embedders.w2v import CW2V

from gensim.models import Word2Vec
from tqdm import tqdm


class ArxivCurie(base.AbstractModel):
    
    __name__ = "ArxivCurie"
    
    def __init__(self):
        super().__init__()
        self._loader = arxiv.ArxivReader()
        self._cleaner = nltk.NLTKCleaner()
        self._embedder = CW2V()

    def online_training(self, query: str, max_results=1000, load_kwargs={}, embed_kwargs={}):
        self._loader.query(query, max_results, **load_kwargs)
        for paper in tqdm(self._loader.retrieve()):
            # extract sentences from paper, and clean
            text = self._cleaner.to_sentence(paper)
            self._embedder.train(text, **embed_kwargs)

    def summarize_pdf(self, path: str) -> str:
        text = self._loader.retrieve()
