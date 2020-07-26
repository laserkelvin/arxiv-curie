
from typing import List
from abc import ABC, abstractmethod

import numpy as np


class AbstractEmbedder(ABC):
    @abstractmethod
    def sentence_embedding(self, text: List[str]) -> np.ndarray:
        """
        Function that will take a sentence in the form of a list,
        and compute the corresponding embedding vector. The result
        should correspond to an array-like object, preferably NumPy.

        Parameters
        ----------
        text : List[str]
            List of words that compose a sentence

        Returns
        -------
        np.ndarray
            A NumPy array corresponding to the sentence embedding
        """
        raise NotImplementedError