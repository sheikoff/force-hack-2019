import math
import pickle

import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

abbr_catalog = pd.read_excel(r'data\abbr.xlsx')


# lit_dict = {}
# for i in range(len(abbr_catalog)):
#     cat = abbr_catalog['category'][i]
#     if cat.lower() == 'lithology':
#         lit_dict[]


def split(txt, seps):
    default_sep = seps[0]
    for sep in seps[1:]:
        txt = txt.replace(sep, default_sep)
    return [i.strip() for i in txt.split(default_sep)]


def process(str):
    tokens = split(str, (',', '.', '-', '/'))
    # remove empty
    tokens = list(filter(None, tokens))

    # print(tokens)

    result = {}
    # unknown = []
    for t in tokens:
        if t:
            found = False
            for i in range(len(abbr_catalog)):
                if t.lower() == abbr_catalog['abbreviation'][i].lower():
                    # print(i, t, abbr_catalog['Long'][i], abbr_catalog['category'][i])
                    result[t] = [abbr_catalog['Long'][i], abbr_catalog['category'][i]]
                    found = True
                    break
            if not found:
                # unknown.append(t)
                result[t] = ['', '']
    return result


# result = process('Sst.Lt-gry.F-gr.Sbang.Uncons.Fr-srt. -')
# print('Original=', tokens)
# print('Result=', result)

#dump of the original file data/RealPore Por Perm Lithology data 1240 Wells Norway public.xlsx
with open('filtered.bin', 'rb') as f:
    data = pickle.load(f)

TEST_COL='main lithology'
#TEST_COL='clean lithology'
#TEST_COL='grain size'

data.dropna(subset=[TEST_COL], inplace=True)

params_data = np.array(data[['Measured Depth',
                             'Permeability horizontal ke  (klinkenberg corrected)  also called kl. KL',
                             'Permeability vertical ka  (air) ',
                             'Porosity helium',
                             'porosity best of available',
                             'gain density gr/cm3']])
desc_data = np.array(data[TEST_COL])

#make all one case
for i in range(len(desc_data)):
    desc_data[i] = desc_data[i].lower()

le = preprocessing.LabelEncoder()
desc_data_num = le.fit_transform(desc_data)
print(desc_data_num)

cc = Counter(desc_data)
print(cc)

def predict():
    X = params_data
    y = desc_data_num

    # cleans data
    for i in range(len(X)):
        for j in range(len(X[i])):
            if isinstance(X[i][j], str):
                X[i][j] = 0
            elif math.isnan(X[i][j]):
                X[i][j] = 0
            elif math.isinf(X[i][j]):
                X[i][j] = np.finfo(np.float64).max

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    model = KNeighborsClassifier(n_neighbors=9)
    model.fit(x_train, y_train)

    acc = model.score(x_test, y_test)
    print(acc)

    predicted = model.predict(x_test)

    print(x_test)

    for i in range(len(predicted)):
        if y_test[i] != predicted[i]:
            print("Predicted: ", predicted[i], "Data: ", x_test[i], "Actual: ", y_test[i])


predict()

#
# print(desc_data)
#
# for i in range(len(desc_data)):
#     result = process(desc_data[i])
#
#     print(result)

# data = pd.read_excel(r"data\RealPore Por Perm Lithology data 1240 Wells Norway public.xlsx")
# #remove empty
# data.dropna(subset=['Formation description original'], inplace=True)
