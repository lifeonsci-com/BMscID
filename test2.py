import numpy as np
import pandas as pd
from sklearn import preprocessing

# CD20RRMM_normal_exp.txt

train_data = pd.read_csv('./data/train_data.txt', sep="	", header=0)
train_data = train_data.transpose()

test_data = pd.read_csv('./data/CD20RRMM_normal_exp.txt', sep="	", header=0)
test_data = test_data.transpose()

Shape = test_data.shape
Row = Shape[0]

print (train_data)

print (test_data)

X = pd.concat([train_data, test_data], axis=0, join='inner')
print (X)





