from curie.models import base
from curie.loaders import arxiv, offline
from curie.cleaners import nltk
from curie.embedders.w2v import CW2V

from tqdm import tqdm


class ArxivCurie(base.AbstractModel):

    __name__ = "ArxivCurie"

    def __init__(self):
        super().__init__()
        self._loader = arxiv.ArxivReader()
        self._cleaner = nltk.NLTKCleaner()
        self._embedder = CW2V()

    def online_training(
        self, query: str, max_results=1000, epochs=5, load_kwargs={}
    ) -> None:
        self._loader.query(query, max_results, **load_kwargs)
        for paper in tqdm(self._loader.retrieve()):
            # extract sentences from paper, and clean
            text = self._cleaner.to_sentence(paper)
            self._embedder.build_vocab(text, update=True)
            self._embedder.train(text, epochs=epochs, total_examples=len(text))

    def offline_training(self, path: str, epochs=3) -> None:
        """
        Function to perform the training routine using downloaded PDFs.
        This function overrides the `Loader` class the model would otherwise
        be using in favor of an `OfflineReader`, that simply loads and parses
        PDFs with `tika`.

        For every paper, we run the text through the cleaning pipeline, then
        pass it to our `Word2Vec` wrapper to train with.

        Parameters
        ----------
        path : str
            [description]
        """
        loader = offline.OfflineReader()
        paths = loader.query(path)
        for paper in tqdm(loader.retrieve(paths)):
            text = self._cleaner.to_sentence(paper)
            self._embedder.build_vocab(text, update=True)
            self._embedder.train(text, epochs=epochs, total_examples=len(text))

    def summarize_pdf(self, path: str) -> str:
        text = self._loader.read_pdf(path)
        text = self._cleaner.to_sentence(text)
        embeddings = self._embedder.sentence_embedding(text)
        return embeddings
