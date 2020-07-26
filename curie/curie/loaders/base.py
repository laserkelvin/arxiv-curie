
"""
base.py

This module defines base classes for readers.
"""

from abc import ABC, abstractmethod


class AbstractLoader(ABC):    
    @abstractmethod
    def query(self, search_terms: str, max_results=200, keep=False):
        """
        Base method for querying and where possible downloading papers and
        parsing it into text.

        Parameters
        ----------
        search_terms : str
            [description]

        Raises
        ------
        NotImplementedError
            [description]
        """
        raise NotImplementedError

    def clean_text(self, *cleaners) -> str:
        """
        This function implements a text cleaning pipeline, where the arguments
        correspond to callable instances of the `Cleaner` class. Each cleaning
        operates on the text in the order they're provided.

        Returns
        -------
        str
            Cleaned text, based on the pipeline designated in `cleaners`.
        """
        text = self.text
        for cleaner in cleaners:
            text = func(text)
        return text