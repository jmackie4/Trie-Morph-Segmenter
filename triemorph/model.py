import os,re
from typing import List
from nltk.util import flatten
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import triemorph.tokenization_pipeline as tp
from collections import Counter

class TrieNode:
    def __init__(self,key,value=None,parent=None):
        self.key = key
        self.value = value
        self.children = {}
        self.parent = parent
        self.is_end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode(' ')

    def _add_word_recursively(self,node,word,i):
        if i < len(word):
            char = word[i]
            if char not in node.children:
                node.children[char] = TrieNode(char,parent=node)
            print(node.children[char].key)
            self._add_word_recursively(node.children[char],word,i+1)

        else:
            node.is_end_of_word = True

    def add_word(self,word):
        self._add_word_recursively(self.root,word,0)


    def _search_recursively(self,node,word,i,len):
        if node is None:
            return False

        if node.is_end_of_word == True and i == (len-1):
            return True

        char = word[i]
        return self._search_recursively(node.children[char],word,i+1,len)

    def search(self,word):
        if self._search_recursively(self.root,word,0,len(word)+1):
            print(f'{word} found in the trie!!')

        else:
            print(f'{word} not found in the trie!!')

class Entropy_Node(TrieNode):
    def __init__(self,key,value=None,parent=None):
        super().__init__(key,value,parent)
        self.entropy = None
        self.child_counts = Counter()


class Entropy_Trie(Trie):
    def __init__(self):
        super().__init__()
        self.root = Entropy_Node(' ')

    def _add_word_recursively(self,node,word,i):
        if i < len(word):
            char = word[i]
            if char not in node.children:
                node.children[char] = Entropy_Node(char,parent=node)
                node.child_counts[char] += 1
            print(node.children[char].key)
            self._add_word_recursively(node.children[char],word,i+1)

        else:
            node.is_end_of_word = True

    def add_word(self,word):
        self._add_word_recursively(self.root,word,0)

    def calculate_entropy_recursively(self, node):
        if len(node.children) == 0:
            return 0
        else:
            initial_series = pd.Series(node.child_counts.values())
            probability_vector = initial_series.div(initial_series.sum())
            entropy = -np.dot(probability_vector, np.log2(probability_vector))
            node.entropy = entropy
            for child in node.children.values():
                self.calculate_entropy_recursively(child)

    def fill_entropies(self):
        self.calculate_entropy_recursively(self.root)




class Trie_Model(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass

    def fit(self,X,y=None):
        return self

    def transform(self,X:List[str],y=None):
        trie = Trie(X)
        for word in set(X):
            trie.add_word(word)
        return trie


def process_corpus(func):
#This is a decorator for the create trie function that creates a trie when given a folder of texts
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
        trie = func(os.path.join(cwd, corpus_name), tokenizer_pattern)
        print('You now have fully loaded the corpus into a trie, have fun!!!')
        return trie
    return wrapper


def create_trie(path:str,pattern:str):
    assert os.path.exists(path), 'You can only create a trie using a legitimate path to a corpus!'
    pipeline_steps = [('tokenize',tp.CorpusTokenizer(pattern=pattern)),
                      ('trie',Trie_Model())]
    pipeline = Pipeline(pipeline_steps)
    return pipeline.fit_transform(path)




