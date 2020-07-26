
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
    """
    def __call__(self, text: Union[str, List[str]]) -> str:
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