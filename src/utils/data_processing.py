import pandas as pd
import numpy as np

import src.utils.project_paths as ProjectPaths

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_regression

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

def plotHistogram(df, columnname):
    return sns.displot(df[columnname], kde=True, height=4, aspect=2)

def encodeDataForMI(df):
    """
    Removing NaN and encoding categorical features into a numeric representation.
    Filling NaN in numeric features with the median value of the column.
    :param df:
    :return:
    """
    numerical_features = df.select_dtypes(include=[np.number])
    categorical_features = df.select_dtypes(include=['object', 'category'])

    for column in numerical_features:
        df[column].fillna(df[column].median(), inplace=True)

    for column in categorical_features:
        df[column], _ = df[column].factorize()

def getMIScore(X, Y):
    score = mutual_info_regression(X, Y)
    features = pd.Series(score, name="MI Score", index=X.columns).sort_values(ascending=False)
    return features

def plotMIScore(score, N=20):
    top_features = score.head(N)

    plt.figure(figsize=(10, 7))
    sns.barplot(x=top_features.values, y=top_features.index, orient='h')

    plt.title(f"Mutual Information Scores for Top {N} Features")
    plt.xlabel("Mutual Information Score")
    plt.ylabel("Features")

    for index, value in enumerate(top_features.values):
        plt.text(value, index, f"{value:.2f}", ha='left', va='center')

    return plt.show()

def plotScatterForFeatures(df, columnX, columnY):

    data = pd.concat([df[columnY], df[columnX]], axis=1)
    sns.regplot(x=columnX, y=columnY, data=data, scatter_kws={'s':50, 'alpha':0.5}, line_kws={'color':'orange'})

    plt.title(f"{columnX} with {columnY}")
    plt.xlabel(columnX)
    plt.ylabel(columnY)

    return plt.show()


def plotBoxPlotForFeatures(df, columnX, columnY):
    data = pd.concat([df[columnY], df[columnX]], axis=1)
    f, ax = plt.subplots(figsize=(16, 8))
    fig = sns.boxplot(x=columnX, y=columnY, data=data)
    fig.axis(ymin=0, ymax=800000)
    plt.xticks(rotation=90)
    return plt.show()

def plotCorrelationMatrix(df, columnY, N=10):
    matrix = df.corr(numeric_only=True)
    columns = matrix.nlargest(N, columnY,)[columnY].index
    correlation_matrix = np.corrcoef(df[columns].values.T)
    sns.set(font_scale=1.25)
    heatmap = sns.heatmap(correlation_matrix, cbar=True, annot=True, annot_kws={'size': N}, square=True, fmt='.2f',
                          yticklabels=columns.values, xticklabels=columns.values)

    return plt.show()

def writeOutput(Y, path, offset=1461):
    with open(path, 'w') as handle:
        handle.write(f"Id,SalePrice\n")
        for index, predict in enumerate(Y):
            handle.write(f"{offset+index},{predict[0]}\n")
