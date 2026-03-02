import unittest
import triemorph.morph_segmenter as morph
import numpy as np
import pandas as pd
import triemorph.model as model
from collections import defaultdict,Counter

class ArrayMappingMakerTest(unittest.TestCase):
    def setUp(self):
        self.test_wordlist = ['the','cat','in','the','hat','once','upon','a','time','there','was',
            'a','tiny','little','skibidi','man','with','a','tiny','skiboodle','plan',
            'the','skibidi','man','takes','his','skibini','plan','to','the','weird','why',
            'man']
        self.test_wordlist2 = [i for i in range(30)]

    def test_load_wordlist(self):
        output = morph.load_model_with_wordlist(self.test_wordlist)
        self.assertIsInstance(output,defaultdict)
        self.assertTrue(all(value != {} for value in output.values()))

    def test_load_wordlist3(self):
        with self.assertRaises(TypeError):
            morph.load_model_with_wordlist(self.test_wordlist2)

    def test_array_mapping_maker_init(self):
        output = morph.ArrayMappingMaker(self.test_wordlist)
        self.assertIsInstance(output,morph.ArrayMappingMaker)

    def test_create_arrays_and_mappings(self):
        array_mapping_maker = morph.ArrayMappingMaker(self.test_wordlist)
        output = array_mapping_maker.create_arrays_and_mappings()
        self.assertTrue(len(output) == len(array_mapping_maker.model))
        self.assertTrue(all(isinstance(value,tuple) for value in output.values()))

    def test_invalid_input_type(self):
        with self.assertRaises(TypeError):
            morph.ArrayMappingMaker(self.test_wordlist2)


class SurprisalAssignerTest(unittest.TestCase):
    def setUp(self):
        self.test_wordlist = ['the', 'cat', 'in', 'the', 'hat', 'once', 'upon', 'a', 'time', 'there', 'was',
                              'a', 'tiny', 'little', 'skibidi', 'man', 'with', 'a', 'tiny', 'skiboodle', 'plan',
                              'the', 'skibidi', 'man', 'takes', 'his', 'skibini', 'plan', 'to', 'the', 'weird', 'why',
                              'man']

    def test_create_surprisal_assigner(self):
        array_mapping_maker = morph.ArrayMappingMaker(self.test_wordlist)
        test_arraymappings = array_mapping_maker.create_arrays_and_mappings()
        output = morph.SurprisalAssigner(test_arraymappings)
        self.assertTrue(isinstance(output,morph.SurprisalAssigner))

    def test_assign_specific_surprisal(self):
        array_mapping_maker = morph.ArrayMappingMaker(self.test_wordlist)
        test_arraymappings = array_mapping_maker.create_arrays_and_mappings()
        test_assigner = morph.SurprisalAssigner(test_arraymappings)
        output = test_assigner.assign_specific_surprisal(self.test_wordlist[14],3)
        self.assertTrue(isinstance(output,tuple))
        self.assertIsInstance(output[1],float)

    def test_assign_surprisals(self):
        array_mapping_maker = morph.ArrayMappingMaker(self.test_wordlist)
        test_arraymappings = array_mapping_maker.create_arrays_and_mappings()
        test_assigner = morph.SurprisalAssigner(test_arraymappings)
        output = test_assigner.assign_surprisals(self.test_wordlist[14])
        self.assertTrue(isinstance(output,dict))


class SegmenterTest(unittest.TestCase):
    def setUp(self):
        self.test_wordlist = ['the', 'cat', 'in', 'the', 'hat', 'once', 'upon', 'a', 'time', 'there', 'was',
                              'a', 'tiny', 'little', 'skibidi', 'man', 'with', 'a', 'tiny', 'skiboodle', 'plan',
                              'the', 'skibidi', 'man', 'takes', 'his', 'skibini', 'plan', 'to', 'the', 'weird', 'why',
                              'man']

    def test_create_segmenter(self):
        output = morph.Segmenter(self.test_wordlist)
        self.assertIsInstance(output,morph.Segmenter)

    def test_segment_wordlist(self):
        segmenter = morph.Segmenter(self.test_wordlist)
        output = segmenter.segment_wordlist(self.test_wordlist)
        self.assertTrue(all(isinstance(item,str) for item in output))
        self.assertTrue(all('.' in item for item in output))
        self.assertTrue(len(output) == len(self.test_wordlist))
