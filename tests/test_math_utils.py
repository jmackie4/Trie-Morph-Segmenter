import unittest
import triemorph.math_utils as mutils
import pandas as pd

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


