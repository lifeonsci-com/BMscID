import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.ensemble import RandomForestClassifier

from sklearn.semi_supervised import LabelSpreading
from sklearn.datasets import make_circles

from client_config import *

# Preprocess

le = preprocessing.LabelEncoder()
Label = pd.read_csv(metadataPath, sep="	", index_col=0)
# data = data.to_dict()
train_data = pd.read_csv(traindataPath, sep="	", header=0)
train_data = train_data.transpose()


FinalData = pd.concat([train_data, Label], axis=1)

y = FinalData['CellType']
X = FinalData.drop(['CellType'], axis=1)

smo = SMOTE(random_state=42, k_neighbors=3)
LabelX, Labely = smo.fit_sample(X, y)
le.fit(Labely)
Labely = le.transform(Labely)

# Construct unlabel Y

test_data = pd.read_csv(testdataPath, sep="	", header=0)
test_data = test_data.transpose()

print (test_data)

X = pd.concat([LabelX, test_data], axis=0, join='inner')
print (X)

Features = X.columns.values.tolist()
LabelXFeatures = X.loc[:, Features]


Shape = test_data.shape
Row = Shape[0]

Test = [-1 for i in range(0, Row)]

# y = Labely.extend(Test)

y = np.append(Labely,Test)

print (Labely)
print (Test)
print (y)

# Knn LabelSpreading



label_spread = LabelSpreading(kernel='knn', alpha=0.8)
label_spread.fit(X, y)
output_labels = label_spread.transduction_
score = label_spread.score(LabelXFeatures, Labely)

print (output_labels)
# accuracy
print ("score : ", score)


# Result
# le.inverse_transform(output_labels)


