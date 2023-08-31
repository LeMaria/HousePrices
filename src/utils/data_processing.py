import pandas as pd
import numpy as np

import src.utils.project_paths as ProjectPaths


def loadData():
    train = loadTrainData()
    test = loadTestData()
    return train, test


def loadTrainData():
    return pd.read_csv(ProjectPaths.getTrainPath())

def loadTestData():
    return pd.read_csv(ProjectPaths.getTestPath())

def getSummaryForNumericalFeatures(df):
    numerical_features = df.select_dtypes(include=[np.number])
    return numerical_features.describe().T

def getSummaryForCategoricalFeatures(df):
    categorical_features = df.select_dtypes(include=['object', 'category'])
    return categorical_features.describe().T

def getSummaryForMissingValues(df):
    missing_values = (df.isnull().sum())
    return missing_values[missing_values > 0]


