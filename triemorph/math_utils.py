import pandas as pd
import numpy as np

def create_joint_probability_df(input:pd.DataFrame):
    return input.div(input.sum().sum())

def create_marginal_probability(input:pd.DataFrame,marginal:str='y'):
    total_sum = input.sum().sum()
    if marginal == 'y':
        return input.sum() / total_sum
    else:
        return input.sum(axis=1) / total_sum

def create_conditional_probabilities(input:pd.DataFrame,conditional:str='y'):
    if conditional == 'y':
        output = input.div(input.sum(axis=0),axis=1)
    else:
        output = input.div(input.sum(axis=1),axis=0)
    return output

def get_conditional_entropy(input:pd.DataFrame,conditional:str='x'):
    joint_probability_table = create_joint_probability_df(input)
    log_conditional_probabilities = np.log2(create_conditional_probabilities(input,conditional=conditional)).replace([np.inf, -np.inf], 0)
    return -(joint_probability_table * log_conditional_probabilities).sum().sum()

def get_entropy_for_variable(input:pd.DataFrame,marginal:str='y'):
    marginal_probabilities = create_marginal_probability(input,marginal=marginal)
    return -(np.log2(marginal_probabilities) * marginal_probabilities).sum()

def get_mutual_information(input:pd.DataFrame):
    marginal_entropy = get_entropy_for_variable(input,marginal='x')
    conditional_entropy = get_conditional_entropy(input,conditional='y')
    return marginal_entropy - conditional_entropy