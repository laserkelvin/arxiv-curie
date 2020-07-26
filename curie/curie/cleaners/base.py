
from typing import List, Union

from abc import ABC, abstractmethod, abstractstaticmethod

class AbstractCleaner(ABC):
    @abstractmethod
    def __call__(self, text: str) -> str:
        return self.clean(text)

    @property
    @abstractmethod
    def pipeline(self):
        """
        Property corresponding to an iterable chain of callable functions
        that compose a pipeline of text transformations.
        """
        return self._pipeline

    @abstractstaticmethod
    def tokenize_sentence(text: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def clean(self, text: str) -> str:
        for func in self._pipeline:
            text = func(text)
        return text


class AbstractCleaningMethod(ABC):
    @abstractmethod
    def __call__(self, text):
        raise NotImplementedError


class RemoveCharacters(AbstractCleaningMethod):
    def __init__(self, *characters):
        super().__init__()
        self.characters = characters
    
    def __call__(self, text: str) -> str:
        for character in self.characters:
            text = text.replace(character, "")
        return text


class ReplaceWhitespace(AbstractCleaningMethod):
    """
    Delete typical whitespace characters, including
    newline, tab, carriage return, and vertical tab characters.
    
    This function will return results in the same type as the inputs:
    if a list of strings (i.e. sentences) are provided, a list of
    strings with whitespaces removed will be returned.
    """
    def __call__(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        if type(text) == list:
            return list(map(replace_whitespace, text))
        else:
            return replace_whitespace(text)


def replace_whitespace(text: str) -> str:
    """
    Function to delete typical whitespace characters, including
    newline, tab, carriage return, and vertical tab characters.

    Parameters
    ----------
    text : str
        Input text

    Returns
    -------
    str
        Cleaned text
    """
    for character in ["\n", "\t", "\r", "\v"]:
        text = text.replace(character, "")
    return text


class RemoveArxivUrl(AbstractCleaningMethod):
    """
    This function will return results in the same type as the inputs:
    if a list of strings (i.e. multiple URLs) are provided, a list of
    URLs with the arxiv part removed will be returned.
    """
    def __call__(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        if type(text) == list:
            return list(map(remove_arxiv_url, text))
        else:
            return remove_arxiv_url(text)


def remove_arxiv_url(text: str) -> str:
    """
    Remove the ArXiv URL boilerplate from a string. This is primarily
    for truncating `pdf_url` from ArXiv queries for lightweight
    saving.

    Parameters
    ----------
    text : str
        

    Returns
    -------
    str
        [description]
    """
    return text.replace("http://arxiv.org/pdf/", "")