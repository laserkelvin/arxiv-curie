from abc import ABC, abstractmethod


class AbstractSummarizer(ABC):
    @abstractmethod
    def summarize(self, text: str) -> str:
        """
        Abstract method for summarizing a piece of text. Long text
        goes in, short text comes out, regardless of the higher level
        method being used.

        Parameters
        ----------
        text : str
            Input text to be summarized

        Returns
        -------
        str
            Summarized text.
        """
        raise NotImplementedError

    def __call__(self, text: str) -> str:
        return self.summarize(text)


class ExtractiveSummarizer(AbstractSummarizer):
    """
    Base class for extractive summarizers. This method of summarization
    re-uses text verbatim, and is not generative.
    """

    def __init__(self, n_sentences=10):
        super().__init__()
        self.n_sentences = n_sentences


class AbstractiveSummarizer(AbstractSummarizer):
    """
    Base class for abstractive summarizers. This method of summarization
    is significantly more complex than `ExtractiveSummarizer`, requiring
    a model of sorts that can extract semanticity from text and generate
    new text.

    This is more of a TODO class, and some methods relying on deep learning
    will be implemented.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
