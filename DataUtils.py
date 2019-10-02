import numpy as np
import pandas as pd

def get_iris_data():
    return pd.read_csv("datasets/iris.data", header=None).to_numpy()

def get_column_data_3C():
    dataset = pd.read_csv("datasets/column_3C.data", header=None)
    dataset.columns = ['0', '1', '2', '3', '4', '5', 'class']

    dataset[['0', '1', '2', '3', '4', '5']] = dataset[['0', '1', '2', '3', '4', '5']].apply(normalize)

    return dataset.to_numpy()

def get_column_data_2C():
    dataset = pd.read_csv("datasets/column_2C.data", header=None)
    dataset.columns = ['0', '1', '2', '3', '4', '5', 'class']

    dataset[['0', '1', '2', '3', '4', '5']] = dataset[['0', '1', '2', '3', '4', '5']].apply(normalize)

    return dataset.to_numpy()

def normalize(df):
    return (df-df.min())/(df.max()-df.min())