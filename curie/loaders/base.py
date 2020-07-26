"""
base.py

This module defines base classes for readers.
"""

from abc import ABC, abstractmethod

from tika import parser


class AbstractLoader(ABC):
    @property
    @abstractmethod
    def results(self):
        raise NotImplementedError

    @abstractmethod
    def query(self, query: str, max_results=200, keep=False):
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

    def read_pdf(self, path: str) -> str:
        """
        Use `Tika` parser to extract text from a PDF file. The only thing
        returned with this call is the text, although there are additional
        metadata (that aren't necessarily so useful) returned by `Tika`.

        Parameters
        ----------
        path : str
            Path to a PDF file

        Returns
        -------
        str
            Parsed content of a PDF file
        """
        paper = parser.from_text(path)
        return paper["content"]
