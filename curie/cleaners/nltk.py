from typing import List

from nltk import tokenize

from curie.cleaners import base


class NLTKCleaner(base.AbstractCleaner):
    def __init__(self, pipeline=None):
        super().__init__()
        if pipeline is None:
            pipeline = [base.ReplaceWhitespace()]
        self._pipeline = pipeline

    @property
    def pipeline(self):
        return self._pipeline

    def clean(self, text: str) -> str:
        for func in self._pipeline:
            text = func(text)
        return text

    def tokenize_sentence(self, text: str) -> List[str]:
        text = tokenize.sent_tokenize(text)
        return self.clean(text)

    def tokenize_word(self, text: str) -> List[str]:
        text = tokenize.word_tokenize(text)
        return self.clean(text)
