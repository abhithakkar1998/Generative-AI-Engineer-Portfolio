# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 13:36:51 2025

@author: kthakkara
"""

from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Sample data
fruits = np.array(["Apple", "Banana", "Cherry", "Apple"]).reshape(-1, 1)

fruits

encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(fruits)
print("One-hot encoded array:\n", encoded)
print("Categories:", encoder.categories_)



import pandas as pd

# Sample data
df = pd.DataFrame({'Fruit': ["Apple", "Banana", "Cherry", "Apple"]})
# One-hot encoding using get_dummies
encoded_df = pd.get_dummies(df, columns=['Fruit'])
print(encoded_df)