import unittest
import triemorph.math_utils as mutils
import pandas as pd
import numpy as np

class Probability_Calculations_Test(unittest.TestCase):
    def setUp(self):
        self.frequency_table = pd.DataFrame({'column1':[1,2,3,4],
                                             'column2':[5,6,7,8],
                                             'column3':[9,10,11,12],
                                             })

    def test_create_joint_probability_df(self):
        output = mutils.create_joint_probability_df(self.frequency_table)
        self.assertEqual(output.sum().sum(),1)

    def test_create_x_marginals(self):
        output = mutils.create_marginal_probability(self.frequency_table)
        self.assertEqual(output.equals(self.frequency_table.sum()/self.frequency_table.sum().sum()),True)

    def test_create_y_marginals(self):
        output = mutils.create_marginal_probability(self.frequency_table,marginal='hmm interesting')
        self.assertEqual(output.equals(self.frequency_table.sum(axis=1)/self.frequency_table.sum().sum()),True)

    def test_create_conditional_probabilities(self):
        output = mutils.create_conditional_probabilities(self.frequency_table,conditional='this isnt y!!!')
        self.assertEqual(round(output.sum().sum()),len(self.frequency_table))

    def test_create_conditional_probabilities_y(self):
        output = mutils.create_conditional_probabilities(self.frequency_table)
        self.assertEqual(round(output.sum().sum()),len(self.frequency_table.columns))

    def test_get_conditional_entropy(self):
        y_output = mutils.get_conditional_entropy(self.frequency_table,conditional='y')
        x_output = mutils.get_conditional_entropy(self.frequency_table)
        self.assertTrue(y_output != x_output)

    def test_get_entropy_for_variable(self):
        output = mutils.get_entropy_for_variable(self.frequency_table)
        self.assertTrue(output >= 0)

    def test_get_entropy_for_variable_x(self):
        x_output = mutils.get_entropy_for_variable(self.frequency_table,marginal='x')
        y_output = mutils.get_entropy_for_variable(self.frequency_table)
        self.assertNotEqual(y_output,x_output)

    def test_get_mutual_information(self):
        output = mutils.get_mutual_information(self.frequency_table)
        self.assertTrue(output >= 0)

class Three_Variable_Frequency_Table_Test(unittest.TestCase):
    def setUp(self):
        self.test_initial_dict = {('b','t','oa'):3,
                                  ('v','i','is'):7,
                                  ('w','t','es'):5,
                                  ('j','b','ub'):8,
                                  }

    def test_parse_tuple_keys(self):
        output = mutils.parse_tuple_keys(self.test_initial_dict)
        self.assertTrue(len(output) == 3)
        self.assertTrue(all(item != [] for item in output))

    def test_map_int_to_index_items(self):
        input = [['b','v','w','j'],['t','i','t','b'],['oa','is','es','ub']]
        output = mutils.map_int_to_index_items(input)
        self.assertTrue(all(isinstance(item,dict) for item in output))
        self.assertEqual(len(output),len(input))

    def test_create_empty_frequency_array(self):
        output = mutils.create_empty_frequency_array(self.test_initial_dict)
        self.assertIsInstance(output,np.ndarray)

    def test_create_xyz_array(self):
        output = mutils.create_xyz_array(self.test_initial_dict)
        self.assertIsInstance(output,np.ndarray)
        self.assertTrue(all(dim > 0 for dim in output.shape))
        self.assertTrue(output.ndim == 3)

class CMI_Tests(unittest.TestCase):
    def setUp(self):
        self.initial_array = np.random.randint(0,15,size=(3,3,3))
        self.test_joint_probability = self.initial_array / np.sum(self.initial_array)

    def test_create_pxyz(self):
        output = mutils.create_pxyz_array(self.initial_array)
        self.assertTrue(np.isclose(np.sum(output), 1.0))

    def test_get_conditional(self):
        output1 = mutils.get_conditional_probability(self.test_joint_probability,axis=0)
        output2 = mutils.get_conditional_probability(self.test_joint_probability,axis=1)
        self.assertTrue(np.allclose(np.sum(output1,axis=0),1))
        self.assertTrue(np.allclose(np.sum(output2,axis=1),1))


