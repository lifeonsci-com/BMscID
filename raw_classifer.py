import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from client_config import *
# Preprocess

le = preprocessing.LabelEncoder()
Label = pd.read_csv(metadataPath, sep="	", index_col=0)
# data = data.to_dict()
train_data = pd.read_csv(traindataPath, sep="	", header=0)
# train_data = pd.read_csv('./data/temp_train_data.txt', sep="	", header=0)
train_data = train_data.transpose()


FinalData = pd.concat([train_data, Label], axis=1)

y = FinalData['CellType']
X = FinalData.drop(['CellType'], axis=1)

# data unbalance

print(Counter(y))
# {'Mono': 1826, 'T-II': 1416, 'NK': 545, 'B': 436, 'Pre-B-I': 361, 'Ery': 226, 'HSC': 63, 'BM': 54, 'Pro-B-I': 54, 'Pro-Mye': 45, 'CMP': 41, 'Pro-B-II': 35, 'GMP': 32, 'MEP': 26, 'Platelets': 21, 'Pre-B-III': 19, 'T-I': 16, 'Prog': 12, 'T-III': 5, 'Pre-B-II': 5}


smo = SMOTE(random_state=42, k_neighbors=3)
X, y = smo.fit_sample(X, y)

print(Counter(y))

# X_embedded = TSNE(n_components=2).fit_transform(X)
X_embedded = X

le.fit(y)
y = le.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_embedded, y, test_size=0.3)


print(Counter(y_train))
print(Counter(y_test))


# Classifier

clf = RandomForestClassifier(max_depth=3)

clf.fit(X_train, y_train)

# y_predict = clf.predict(X_test)
res_score = clf.score(X_test, y_test)

# score:  0.7181453085067543
print ("score: ", res_score)




















