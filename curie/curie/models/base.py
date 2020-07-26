
from abc import ABC, abstractmethod


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