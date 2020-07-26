from nltk import tokenize

from curie.curie.cleaners import base


class NLTKCleaner(base.AbstractCleaner):
    def __init__(self, pipeline=None):
        super().__init__()
        if pipeline is None:
            pipeline = [base.ReplaceWhitespace()]
        self._pipeline = pipeline

    def clean(self, text: str) -> str:
        for func in self._pipeline:
            text = func(text)
        return text

    def to_sentence(self, text: str) -> str:
        text = tokenize.sent_tokenize(text)
        return self.clean(text)

    def to_words(self, text: str) -> str:
        text = tokenize.word_tokenize(text)
        return self.clean(text)
