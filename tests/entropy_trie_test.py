import unittest
import triemorph.model as model
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import numpy as np

""" DO NOT RUN THE TEST FILE DIRECTLY IT WILL ONLY REALLY WORK IF YOU RUN IT FROM THE PROJECT ROOT 
DIRECTORY"""

class TrieModelTest(unittest.TestCase):
    def setUp(self):
        self.token = 'cheese'

    def test_add_token(self):
        trie = model.Trie()
        trie.add_word(self.token)
        self.assertTrue(trie._search_recursively(trie.root,self.token,0,len(self.token)+1))

    def test_add_token_false(self):
        trie = model.Trie()
        trie.add_word(self.token)
        with self.assertRaises(KeyError):
            trie._search_recursively(trie.root,'bat',0,len('bat')+1)

    def test_trie_path_count_add_on(self):
        trie = model.Trie()
        trie.add_word(self.token)
        self.assertTrue(trie.path_counts != {})


class EntropyTrieTest(unittest.TestCase):
    def setUp(self):
        self.token = 'cheese'
        self.trie = model.Entropy_Trie()
        self.trie.add_word(self.token)
        self.trie.add_word('test')
        self.trie.add_word('carlton')

    def test_get_entropy(self):
        self.assertNotEqual(self.trie.root.entropy,0)

        self.assertNotEqual(self.trie.root.children['c'].entropy,0)

class WordSegmenterTest(unittest.TestCase):
    def setUp(self):
        self.word_list = ['cheese','weenie','wonderland','carlton','wanderlust']
        self.test_input = {1:[np.random.randint(0,10,size=(3,3,3)),self.word_list[0:3]],
                           2:[np.random.randint(0,10,size=(3,3,3)),self.word_list[1:4]],
                           3:[np.random.randint(0,10,size=(3,3,3)),self.word_list[0:3]],
                           }

    def test_load_words(self):
        output = model.load_words(self.word_list)
        self.assertIsInstance(output,dict)

    def test_extract_array_and_mappings(self):
        output = model.extract_array_and_mappings(self.test_input,2)
        self.assertIsInstance(output,tuple)
        self.assertTrue(len(output) == 4)
















