
from nltk import tokenize

import .base

class NLTKCleaner(AbstractCleaner):
    def __init__(self):
        super().__init__()
        self._pipeline = None

    def clean(self):
        