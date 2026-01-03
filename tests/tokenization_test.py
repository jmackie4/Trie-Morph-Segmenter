import unittest, os
from triemorph.tokenization_pipeline import CorpusTokenizer
from triemorph.model import create_trie


class TestTokenization(unittest.TestCase):
    def test_tokenizer_output(self):
        test_tokenizer = CorpusTokenizer(pattern=r'\S+')
        test_path = os.path.join(os.getcwd(),'tests/test_data/testfile_1.txt')
        self.assertIsInstance(test_tokenizer.fit_transform(test_path),list)


    def test_trie_pipeline_output(self):
        test_pattern = r'\S+'
        test_path = os.path.join(os.getcwd(),'tests/test_data/testfile_1.txt')
        self.assertIsInstance(create_trie(test_path,test_pattern,),list)



if __name__ == '__main__':
    unittest.main()

