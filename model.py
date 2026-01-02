import os,re
from typing import List
from nltk.util import flatten
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

import tokenization_pipeline as tp

class TrieNode:
    def __init__(self,key,value=None,children=None,parent=None):
        self.key = key
        self.value = value
        self.children = children
        self.parent = parent
        self.is_end_of_word = False


class Trie:
    def __init__(self,vocabulary:List[str]):
        self.vocabulary_size = len(set(flatten([list(word) for word in vocabulary])))
        self.empty_character_list = [None] * self.vocabulary_size
        self.root = TrieNode(' ',children = self.empty_character_list)
        self.size = 0

    def _add_word_recursively(self,node,word,i):
        if i < len(word):
            idx = ord(word[i]) - ord('a')
            if node.children[idx] is None:
                node.children[idx] = TrieNode(word[i],children = self.empty_character_list,parent=node)
            print(node.children[idx].key)
            self._add_word_recursively(node.children[idx],word,i+1)

        else:
            node.is_end_of_word = True

    def add_word(self,word):
        self._add_word_recursively(self.root,word,0)


    def _search_recursively(self,node,word,i,len):
        if node is None:
            return False

        if node.is_end_of_word == True and i == (len-1):
            return True

        idx = ord(word[i]) - ord('a')
        return self._search_recursively(node.children[idx],word,i+1,len)

    def search(self,word):
        if self._search_recursively(self.root,word,0,len(word)+1):
            print(f'{word} found in the trie!!')

        else:
            print(f'{word} not found in the trie!!')


class Trie_Model(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass

    def fit(self,X,y=None):
        return self

    def transform(self,X:List[str],y=None):
        trie = Trie(X)
        for word in X:
            trie.add_word(word)
        return trie


def process_corpus(func):
    def wrapper():
        cwd = os.getcwd()
        while True:
            corpus_name = input(f'Please put in the path from {cwd} to corpus:\n')
            if os.path.exists(os.path.join(cwd, corpus_name)):
                break
            else:
                print('That\'s a corpus path that does not exist! Try again!')

        tokenizer_pattern = input('Please provide a tokenizer pattern for your corpus:\n')
        tokenizer_pattern = re.compile(tokenizer_pattern)
        trie = create_trie(os.path.join(cwd, corpus_name), tokenizer_pattern)
        print('You now have fully loaded the corpus into a trie, have fun!!!')
        return trie
    return wrapper


@process_corpus
def create_trie(path:str,pattern:str):
    assert os.path.exists(path), 'You can only create a trie using a legitimate path to a corpus!'
    pipeline_steps = [('tokenize',tp.Corpus_Transformer(tokenizer_pattern=pattern)),
                      ('trie',Trie_Model())]
    pipeline = Pipeline(pipeline_steps)
    return pipeline.fit_transform(path)









