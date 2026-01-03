import unittest, os
from triemorph.model import create_trie, Trie
from triemorph.tokenization_pipeline import CorpusTokenizer

""" DO NOT RUN THE TEST FILE DIRECTLY IT WILL ONLY REALLY WORK IF YOU RUN IT FROM THE PROJECT ROOT 
DIRECTORY"""

class TestTokenization(unittest.TestCase):
    def test_tokenizer_output(self):
        test_tokenizer = CorpusTokenizer(pattern=r'\S+')
        test_path = os.path.join(os.getcwd(),'tests/test_data')
        self.assertIsInstance(test_tokenizer.fit_transform(test_path),list)


    def test_trie_pipeline_output(self):
        test_pattern = r'\S+'
        test_path = os.path.join(os.getcwd(),'tests/test_data')
        self.assertIsInstance(create_trie(test_path,test_pattern,),Trie)



if __name__ == '__main__':
    unittest.main()

