from typing import List

from gensim.models.word2vec_inner import MAX_WORDS_IN_BATCH
from gensim.models import Word2Vec
import numpy as np


class CW2V(Word2Vec):
    """
    Wrapper for the Word2Vec model in `gensim`. The only modifications
    made with this class are functions that try to homogenize the API
    in terms of the callables, so that other embedding models can be
    used in the same way (a la `sklearn`)
    """

    def __init__(
        self,
        sentences=None,
        corpus_file=None,
        size=300,
        alpha=0.025,
        window=5,
        min_count=5,
        max_vocab_size=None,
        sample=0.001,
        seed=1,
        workers=3,
        min_alpha=0.0001,
        sg=0,
        hs=0,
        negative=5,
        ns_exponent=0.75,
        cbow_mean=1,
        hashfxn=hash,
        iter=5,
        null_word=0,
        trim_rule=None,
        sorted_vocab=1,
        batch_words=MAX_WORDS_IN_BATCH,
        compute_loss=False,
        callbacks=(),
        max_final_vocab=None,
    ):
        super().__init__(
            sentences=sentences,
            corpus_file=corpus_file,
            size=size,
            alpha=alpha,
            window=window,
            min_count=min_count,
            max_vocab_size=max_vocab_size,
            sample=sample,
            seed=seed,
            workers=workers,
            min_alpha=min_alpha,
            sg=sg,
            hs=hs,
            negative=negative,
            ns_exponent=ns_exponent,
            cbow_mean=cbow_mean,
            hashfxn=hashfxn,
            iter=iter,
            null_word=null_word,
            trim_rule=trim_rule,
            sorted_vocab=sorted_vocab,
            batch_words=batch_words,
            compute_loss=compute_loss,
            callbacks=callbacks,
            max_final_vocab=max_final_vocab,
        )

    def sentence_embedding(self, text: List[str]) -> np.ndarray:
        """
        Function that will take a sentence in the form of a list,
        and compute the corresponding embedding vector. Returns a NumPy
        1D array, corresponding to the embedding vector of the sentence.

        Parameters
        ----------
        text : List[str]
            List of words that compose a sentence

        Returns
        -------
        np.ndarray
            A NumPy array corresponding to the sentence embedding
        """
        vectors = np.vstack([self.wv.get_vector(word) for word in text])
        return np.average(vectors, axis=1)

    def process_sentences(self, text: List[List[str]]) -> List[np.ndarray]:
        """
        Computes embedding vectors for a list of sentences (which in itself
        are lists of words).

        Parameters
        ----------
        text : List[List[str]]
            Nested lists of words within lists of sentences

        Returns
        -------
        List[np.ndarray]
            Sentence vectors for every sentence
        """
        vectors = list(map(self.sentence_embedding, text))
        return vectors
