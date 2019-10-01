import math
import pickle
import clean

import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

abbr_catalog = pd.read_excel(r'data\abbr.xlsx')

#drop all non
# abbr_catalog.dropna(subset=['category'], inplace=True)

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
    tokens = split(str, (',', '.', '-', '/', ' ',')','(',']','['))
    # remove empty
    tokens = list(filter(None, tokens))

    print(tokens)

    result = []
    # unknown = []
    for t in tokens:
        if t:
            found = False
            for i in range(len(abbr_catalog)):
                if t.lower() == abbr_catalog['abbreviation'][i].lower():
                    # print(i, t, abbr_catalog['Long'][i], abbr_catalog['category'][i])
                    result.append([t, abbr_catalog['Long'][i], abbr_catalog['category'][i]])
                    found = True
                    break
            if not found:
                # unknown.append(t)
                result.append([t, '', ''])
    return result

def fixAA(data):
    print(data)
    prev = ''
    for i in range(len(data)):
        s = data[i]
        s = s.strip()
        if s.startswith('a.a.'):
            s = str.join('', (prev, s[4:]))
            data[i] = s
            # print(s)
        prev = s



def process2(input):
    res = []
    prev = None
    changed = False
    prev_word = ''
    for el in input:
        if prev:
            if el[2] == prev[2] and el[2] not in ['accessory']:
                if prev_word != el[1]:
                    prev[1] = str.join(' ', (prev[1], el[1]))
                prev_word = el[1]
                prev[0] = str.join(' ', (prev[0], el[0]))
                changed = True
                continue
            if prev[2] in ['adjective', 'special'] and el[2] not in ['rounding']:
                if prev_word != el[1]:
                    prev[1] = str.join(' ', (prev[1], el[1]))
                prev[2] = el[2]
                prev[0] = str.join(' ', (prev[0], el[0]))
                prev_word = el[1]
                changed = True
                continue
        prev = el
        prev_word = el[1]
        res.append(prev)
    return res, changed


def process3(input):
    for el in input:
        prev = ''
        rr = []
        for e in el[1].split(' '):
            if prev != e:
                rr.append(e)
            prev = e
        el[1] = str.join(' ', rr)
    return input

#dump of /data/dict.xlsx
desc = clean.loadFromDump('desc.bin')

np_desc = np.array(desc['Formation description original'])

np_desc = np_desc[:20]

fixAA(np_desc)

for el in np_desc:
    print(el)
    res = process(el)
    print(res)
    while True:
        res, changed = process2(res)
        if not changed:
            break
    res = process3(res)
    print(res)

# fixAA(np.array(desc['Formation description original']))

# result = process('Sst.Lt-gry.F-gr.Sbang.Uncons.Fr-srt. -')
# print('Result=', result)



# with open('filtered.bin', 'rb') as f:
#     data = pickle.load(f)
#
# TEST_COL='main lithology'
#TEST_COL='clean lithology'
#TEST_COL='grain size'

# data.dropna(subset=[TEST_COL], inplace=True)
#
# params_data = np.array(data[['Measured Depth',
#                              'Permeability horizontal ke  (klinkenberg corrected)  also called kl. KL',
#                              'Permeability vertical ka  (air) ',
#                              'Porosity helium',
#                              'porosity best of available',
#                              'gain density gr/cm3']])
# desc_data = np.array(data[TEST_COL])
#
# #make all one case
# for i in range(len(desc_data)):
#     desc_data[i] = desc_data[i].lower()
#
# le = preprocessing.LabelEncoder()
# desc_data_num = le.fit_transform(desc_data)
# print(desc_data_num)
#
# cc = Counter(desc_data)
# print(cc)
#
# def predict():
#     X = params_data
#     y = desc_data_num
#
#     # cleans date
#     for i in range(len(X)):
#         for j in range(len(X[i])):
#             if isinstance(X[i][j], str):
#                 X[i][j] = 0
#             elif math.isnan(X[i][j]):
#                 X[i][j] = 0
#             elif math.isinf(X[i][j]):
#                 X[i][j] = np.finfo(np.float64).max
#
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
#
#     model = KNeighborsClassifier(n_neighbors=9)
#     model.fit(x_train, y_train)
#
#     acc = model.score(x_test, y_test)
#     print(acc)
#
#     predicted = model.predict(x_test)
#
#     print(x_test)
#
#     for i in range(len(predicted)):
#         if y_test[i] != predicted[i]:
#             print("Predicted: ", predicted[i], "Data: ", x_test[i], "Actual: ", y_test[i])
#
#
# predict()
#
# #
# # print(desc_data)
# #
# # for i in range(len(desc_data)):
# #     result = process(desc_data[i])
# #
# #     print(result)
#
# # data = pd.read_excel(r"data\RealPore Por Perm Lithology data 1240 Wells Norway public.xlsx")
# # #remove empty
# # data.dropna(subset=['Formation description original'], inplace=True)
