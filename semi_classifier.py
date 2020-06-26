import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.semi_supervised import LabelSpreading
from sklearn.model_selection import StratifiedKFold
from client_config import *

# conda create --name myenv python=3.5
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

# test_data = pd.read_csv("./data/tempData.csv", sep=",", header=0, index_col=0)
test_data = pd.read_csv(testdataPath, sep="	", header=0)
test_data = test_data.transpose()

X_train, X_test, y_train, y_test = train_test_split(LabelX, Labely, test_size=0.1, random_state=42)
skf = StratifiedKFold(n_splits=100)
skf.get_n_splits(X, y)

def LabelData(LabelX, unLabelX, Labely, unLabely, testX, testy, batch_id=0):
    LabelXLen = LabelX.shape[0]

    print ("LabeledCellNames", LabelX)

    X = pd.concat([LabelX, unLabelX], axis=0, join='inner')
    y = np.append(Labely,unLabely)

    Features = X.columns.values.tolist()
    testX = testX.loc[:, Features]


    # Knn LabelSpreading
    label_spread = LabelSpreading(kernel='knn', alpha=0.8, max_iter=5)

    label_spread.fit(X, y)
    output_labels = label_spread.transduction_
    score = label_spread.score(testX, testy)

    output_labels = le.inverse_transform(output_labels)
    CellNames = X.index.values.tolist()
    CellResult = {"CellName": CellNames[LabelXLen+1:], "CellType": output_labels[LabelXLen+1:]}

    # Result = pd.DataFrame(data=CellResult)
    # Result.to_csv("./result/%d.csv"%(batch_id), columns=['CellName','CellType'], index=False)

    # accuracy
    print ("score : ", score)

    PredictY = label_spread.predict(testX)
    PredictYLabels = le.inverse_transform(PredictY)
    TrueYLabels = le.inverse_transform(testy)
    PredictResult = {"trueLabel": TrueYLabels, "predictLabel": PredictYLabels}

    # PredictResult = pd.DataFrame(data=PredictResult)
    # PredictResult.to_csv("./result/predict_result_%d.csv"%(batch_id), columns=['trueLabel','predictLabel'], index=False)


    return CellResult



    # le.inverse_transform(output_labels)

# 20 split_part
def split_data(data, split_part):
    Limit = len(data)
    batch_size = int(Limit / split_part)
    datasets = []
    for batch_id in range(split_part):
        temp = data[batch_size * batch_id: (batch_id+1)*batch_size+1]
        datasets.append(temp)
    return datasets

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

Shape = test_data.shape
Row = Shape[0]

unLabelXs = test_data
unLabelys = [-1 for i in range(0, Row)]


unLabelXs_batchsets = split_data(unLabelXs, split_part)
unLabelys_batchsets = split_data(unLabelys, split_part)


Len = len(unLabelXs_batchsets)


Result = pd.DataFrame()
for batch_id in range(Len):
    unLabelX = unLabelXs_batchsets[batch_id]
    unLabely = unLabelys_batchsets[batch_id]
    BatchCellResult = LabelData(X_train, unLabelX, y_train, unLabely, X_test, y_test, batch_id)
    BatchCellResult = pd.DataFrame(data=BatchCellResult)
    # CellResult = merge_two_dicts(CellResult, BatchCellResult)
    Result = Result.append(BatchCellResult)

Result.to_csv("./result/GSE116256_RAW/result.csv", columns=['CellName','CellType'], index=False)



