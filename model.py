from typing import List
from nltk.util import flatten


class TrieNode:
    def __init__(self,key,value=None,children=None):
        self.key = key
        self.value = value
        self.children = children
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
                node.children[idx] = TrieNode(word[i],children = self.empty_character_list)
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




