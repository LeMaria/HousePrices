import src.utils.data_processing as dp

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer


class Processing(object):
    def __init__(self):

        train, test = dp.loadData()
        self.train = train.set_index('Id')
        self.test = test.set_index('Id')

        self.missing_values_columns = dp.getSummaryForMissingValues(train)
        self.missing_values_columns = self.missing_values_columns[self.missing_values_columns > 10].index

        # columns observed from the data visualisation
        self.high_corr_columns = ['GarageArea', 'TotRmsAbvGrd', '1stFlrSF']

        self._filterColumns(self.train)
        self._filterColumns(self.test)

    def _filterColumns(self, df):
        # drop missing value columns
        df.drop(self.missing_values_columns, axis=1, inplace=True)
        # drop highly correlated columns
        columns = list(filter(lambda col: col in df.columns, self.high_corr_columns))
        df.drop(columns, axis=1, inplace=True)

    @staticmethod
    def getPipeline(df):
        """
        returns preprocessing Pipeline
        :param df: dataframe
        :return:
        """
        # in numerical features replace missing values with median and standardize values
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy="median")),
            ('scaler', StandardScaler())
        ])

        # in categorical features replace missing values with most frequent occurring value and transform to one shot format
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])

        categorical_features = df.select_dtypes(include=["object", "category"]).columns
        numerical_features = df.select_dtypes(include=[np.number]).columns

        numerical_cols = numerical_features
        categorical_cols = categorical_features

        # combine both transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols),
            ], remainder="passthrough"
        )

        return Pipeline(steps=[('preprocessor', preprocessor)])
