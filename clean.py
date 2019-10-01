import pandas as pd
import xlsxwriter
import pickle

from pip import __main__


def loadOriginalData():
    data = pd.read_excel(r"data\RealPore Por Perm Lithology data 1240 Wells Norway public.xlsx")
    data = data[['Formation description original','Non sorted Transcription']]
    data.dropna(subset=['Formation description original'], inplace=True)
    return data

def dumpData(name, data):
    with open(name, 'wb') as f:
        pickle.dump(data, f)

def loadFromDump(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def writeToExcel(name, data):
    writer = pd.ExcelWriter(name, engine='xlsxwriter')
    data.to_excel(writer, sheet_name='Sheet1')
    writer.save()

if __name__ == __main__:
    data = loadFromDump('desc.bin')
    writeToExcel('dict.xlsx', data)