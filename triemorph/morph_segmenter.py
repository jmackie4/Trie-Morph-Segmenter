import triemorph.model as model
import pandas as pd
import numpy as np
import triemorph.math_utils as mutils

class Segmenter:
    def __init__(self,wordlist):
        self.array_mapping_maker = ArrayMappingMaker(wordlist)
        self.surprisal_assigner = SurprisalAssigner(self.array_mapping_maker.create_arrays_and_mappings())

    def segment_word(self, word):
        surprisal_results = self.surprisal_assigner.assign_surprisals(word)
        surprisal_series = pd.Series({i + 1: subitem[1] for i, subitem in enumerate(surprisal_results[word])})
        highest_surprisal = surprisal_series.idxmax()
        return word[:highest_surprisal] + '.' + word[highest_surprisal:]

    def segment_wordlist(self, wordlist):
        return [self.segment_word(word) for word in wordlist]


class ArrayMappingMaker:
    def __init__(self,wordlist):
        self.model = load_model_with_wordlist(wordlist)

    def create_arrays_and_mappings(self):
        return {key:mutils.create_xyz_array(value) for key,value in self.model.items()}

class SurprisalAssigner:
    def __init__(self,arrays_and_mappings):
        self.arrays_and_maps = arrays_and_mappings

    def assign_surprisals(self, word):
        j = 1
        surprisals = []
        while j <= len(word):
            surprisals.append(self.assign_specific_surprisal(word, j))
            j += 1
        return {word: surprisals}

    def assign_specific_surprisal(self, word, j):
        x_maps, y_maps, z_maps = self.arrays_and_maps[j][1]
        current_array = self.arrays_and_maps[j][0]
        x_int = x_maps[word[0]]
        z_int = z_maps[word[1:j]]
        if j == len(word):
            y_int = y_maps['$']
            return '$', self.calculate_surprisal(current_array, [x_int, y_int, z_int])
        else:
            y_int = y_maps[word[j]]
            return word[j], self.calculate_surprisal(current_array, [x_int, y_int, z_int])

    def calculate_surprisal(self,array,indices_list):
        z_slice = array[:,:,indices_list[2]]
        pz = np.sum(z_slice,axis=(0,1),keepdims=True)
        pxyz = array[indices_list[0],indices_list[1],indices_list[2]]
        return -np.log2(pxyz/pz).item()

def load_model_with_wordlist(wordlist):
  trie = model.Trie()
  for word in wordlist:
    trie.add_word(word)
  return trie.path_counts