import math
import pickle

import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

def cleanDataFunc(arr):
    for i in range(len(arr)):
        if isinstance(arr[i], str):
            arr[i] = 0
        elif math.isnan(arr[i]):
            arr[i] = 0
        elif math.isinf(arr[i]):
            arr[i] = np.finfo(np.float64).max


abbr_catalog = pd.read_excel(r'data\abbr.xlsx')

#dump of the original file data/RealPore Por Perm Lithology data 1240 Wells Norway public.xlsx
with open('filtered.bin', 'rb') as f:
    data = pickle.load(f)

# for col in data.columns:
#     print(col)

clean_data = data[['clean lithology',
                            'color',
                            'grain size',
                            'rounding',
                            'cement',
                            'sorting',
                            'sed structures',
                            'auxilaries',
                            'auxilaries.1',
                            'auxilaries.2',
                            'auxilaries.3',
                            'auxilaries.4',
                            'auxilaries.5']]


np_clean_data = np.array(clean_data)

for i in range(len(np_clean_data)):
    for j in range(len(np_clean_data[i])):
        if not isinstance(np_clean_data[i][j], str):
            np_clean_data[i][j] = ''

lenc = preprocessing.LabelEncoder()

#depth
np_depth = np.array(data['Measured Depth'])
cleanDataFunc(np_depth)

arr = [np_depth]
for i in range(len(np_clean_data[0,:])):
    t = lenc.fit_transform(np_clean_data[:, i])
    arr.append(t)

np_trans_data = np.vstack(tuple(arr))
np_trans_data = np_trans_data.transpose()

#porosiry
np_porosity = np.array(data['porosity best of available'])
cleanDataFunc(np_porosity)

X = np_trans_data
y = np_porosity

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

model = linear_model.SGDRegressor()
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)
