from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus.reader import PlaintextCorpusReader
from nltk.lm.preprocessing import flatten
from nltk import RegexpTokenizer
from typing import List

class Corpus_Transformer(BaseEstimator,TransformerMixin):
    def __init__(self,tokenizer_pattern=None):
        self._tokenizer_pattern = tokenizer_pattern


    def fit(self,X:str,y=None):
        self._has_tokenizer = False
        if self._tokenizer_pattern:
            self.tokenizer_ = Corpus_Tokenizer.fit(self._tokenizer_pattern)
            self._has_tokenizer = True
        else:
            pass

        return self

    def transform(self,X:str):
        corpus = PlaintextCorpusReader(X,'.*\.txt')
        sents = [corpus.sents(fileid) for fileid in corpus.fileids()]
        if self._has_tokenizer:
            return list(flatten(self.tokenizer_.transform(sents)))
        else:
            return list(flatten(sents))


class Corpus_Tokenizer(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass

    def fit(self,X:str,y=None):
        self.tokenizer_ = RegexpTokenizer(X)
        return self

    def transform(self,X:List[str]):
        return [self.tokenizer_.tokenize(sent) for sent in X]




