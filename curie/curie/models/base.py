
from abc import ABC, abstractmethod

from curie.curie.loaders import offline

from tika import parser
from joblib import dump, load
from tqdm import tqdm


class AbstractModel(ABC):
    """
    This is an abstract class representing a "Curie" model; one can
    think of this as a predefined recipe/pipeline/workflow that
    performs all of the functionality, from reading in text to making
    predictions.
    """
    
    @property
    @abstractmethod
    def name(self):
        return self.__name__
    
    @property
    @abstractmethod
    def loader(self):
        """
        This property corresponds to a `Loader` class that is responsible
        for pulling papers.
        """

    @property
    @abstractmethod
    def cleaner(self):
        """
        This property corresponds to the `Cleaner` class that is responsible
        for taking raw string input, and preparing it for training/evaluation.
        """

    @property
    @abstractmethod
    def summarizer(self):
        """
        This property corresponds to a `Summarizer`, which takes cleaned text
        and uses it to train/evaluate.
        """

    def save(self, path=None):
        if not path:
            path = self.name
        if ".pkl" not in path:
            path += ".pkl"
        dump(self, path)
        print(f"Saved model to {path}.")

    def offline_training(self, path: str):
        loader = offline.OfflineReader()
        paths = loader.query(path)
        for paper in tqdm(loader.retrieve(paths)):
            text = self._cleaner.to_sentence(paper)
            self._embedder.train(text, **embed_kwargs)

    def summarize_pdf(self, path: str) -> str:
        text = parser.from_file(path)["content"]
        # generate sentences and clean
        text = self._cleaner.to_sentence(text)
        vectors = self._embedder.sentence_embedding(text)
        summary = self._summarizer.predict(vectors)
        return summary