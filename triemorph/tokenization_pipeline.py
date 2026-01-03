import re
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus.reader import PlaintextCorpusReader
from nltk.lm.preprocessing import flatten
from nltk import RegexpTokenizer
from typing import List

class CorpusTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self,pattern:str=None):
        self._pattern = re.compile(pattern)

    def fit(self,X,y=None):
        if self._pattern is not None:
            self.tokenizer_ = RegexpTokenizer(self._pattern)
        else:
            pass
        return self

    def transform(self,X:str) -> List[str]:
        corpus = PlaintextCorpusReader(X,r'.*\.txt',word_tokenizer=self.tokenizer_)
        return [token.lower() for token in list(flatten(corpus.sents(corpus.fileids())))]








