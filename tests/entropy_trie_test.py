import unittest
import triemorph.model as model
import pandas as pd


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

