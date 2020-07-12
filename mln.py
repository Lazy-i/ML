#!/user/bin/env python
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import math

income = pd.read_csv("train.csv")
income_test = pd.read_table('test.csv', sep=',')

columns = ['id', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country',
           'income']
columns_test = ['id', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
for name in columns:
    col = pd.Categorical(income[name])
    income[name] = col.codes

for name in columns_test:
    col_test = pd.Categorical(income_test[name])
    income_test[name] = col_test.codes

entropy = -(2/5 * math.log(2/5, 2) + 3/5 * math.log(3/5, 2))
prob_0 = income[income["high_income"] == 0].shape[0] / income.shape[0]
prob_1 = income[income["high_income"] == 1].shape[0] / income.shape[0]
income_entropy = -(prob_0 * math.log(prob_0, 2) + prob_1 * math.log(prob_1, 2))

def calc_entropy(column):
    """
    Calculate entropy given a pandas Series, list, or numpy array.
    """
    # Compute the counts of each unique value in the column.
    counts = np.bincount(column)
    # Divide by the total column length to get a probability.
    probabilities = counts / len(column)

    # Initialize the entropy to 0.
    entropy = 0
    # Loop through the probabilities, and add each one to the total entropy.
    for prob in probabilities:
        if prob > 0:
            entropy += prob * math.log(prob, 2)

    return -entropy

income_entropy = calc_entropy(income["income"])
median_age = income["age"].median()

left_split = income[income["age"] <= median_age]
right_split = income[income["age"] > median_age]

age_information_gain = income_entropy - ((left_split.shape[0] / income.shape[0]) * calc_entropy(left_split["high_income"]) + ((right_split.shape[0] / income.shape[0]) * calc_entropy(right_split["high_income"])))

def calc_information_gain(data, split_name, target_name):
    """
    Calculate information gain given a dataset, column to split on, and target.
    """
    # Calculate original entropy.
    original_entropy = calc_entropy(data[target_name])

    # Find the median of the column we're splitting.
    column = data[split_name]
    median = column.median()

    # Make two subsets of the data based on the median.
    left_split = data[column <= median]
    right_split = data[column > median]

    # Loop through the splits, and calculate the subset entropy.
    to_subtract = 0
    for subset in [left_split, right_split]:
        prob = (subset.shape[0] / data.shape[0])
        to_subtract += prob * calc_entropy(subset[target_name])

    # Return information gain.
    return original_entropy - to_subtract

# Verify that our answer is the same as in the last screen.
columns = ["age", "workclass", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "hours_per_week", "native_country"]
information_gains = []
# Loop through and compute information gains.
for col in columns:
    information_gain = calc_information_gain(income, col, "high_income")
    information_gains.append(information_gain)

# Find the name of the column with the highest gain.
highest_gain_index = information_gains.index(max(information_gains))
highest_gain = columns[highest_gain_index]

