import pandas as pd
import numpy as np


train = pd.read_json('data/train.json')
test = pd.read_json('data/test.json')

print(train.shape)
print(train.head)

print(test.shape)
print(test.head)

