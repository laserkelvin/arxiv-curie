"""
base.py

This module provides the base classes for all the different methods of
embedding generation, semantic extraction, and summarization.

The idea is to organize different methodologies into submodules, for example
in a `word2vec` module that wraps `gensim.models.word2vec`for our purposes.
"""


class Curie(object):
    """
    Main interaction class for `arxiv-curie`. Basically combines the three
    components: a reader, embedder, and a summarizer, to provide a single
    interface that controls the source of papers, how its digested, and
    finally how its summarized.

    Parameters
    ----------
    object : [type]
        [description]
    """

    def __init__(self, reader, embedder, summarizer):
        super().__init__()
        self._reader = reader
        self._embedder = embedder
        self._summarizer = summarizer

    def summarize(self, n_sentences=10):
        return self.summary[:n_sentences]
