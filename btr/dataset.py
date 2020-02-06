import pandas as pd
import numpy as np
import re


class Dataset(object):
    def __init__(self, settings, usecols=None, cols_only=False,
                 transform_dataset=True):
        self.settings = settings
        self.name = self.settings['name']
        self.filepath = self.settings['filepath']
        self.meta_cols = self.settings['meta_columns']
        self.target = self.settings['target']
        self.id_col = self.settings['ID_column']
        self.confounder = self.settings.get('confounder')
        self.data_cols = []
        self.data = pd.DataFrame()
        self._load_data(usecols=usecols)
        self._get_data_cols()
        if transform_dataset:
            self.transform_dataset()
        self.annotations = {}

    def sample_data_cols(self, k, seed=None):
        """ Select a random sample of k-items from a list of columns.
        """
        data_cols = self.data_cols
        prng = np.random.RandomState(seed)
        sampled_cols = prng.choice(data_cols, k, replace=False)
        return [x for x in data_cols if x in sampled_cols]

    def get_X_y(self, test_ids, column=None, selected_cols=None):
        """ Given a dataframe, a target col, and a sample of data column names,
        generate X, a numpy array of the data, and y, a list of target values
        corresponding to each row.
        """
        df = self.data
        df[self.id_col] = df[self.id_col].astype("category")
        if not column:
            column = self.id_col
        if not selected_cols:
            selected_cols = self.data_cols
        df_train, df_test = get_train_test_df(df, test_ids, column)
        df_train[self.id_col].cat.set_categories(test_ids, inplace=True)
        X_train = df_train[selected_cols].to_numpy()
        y_train = df_train[self.target].tolist()
        X_test = df_test[selected_cols].to_numpy()
        y_test = df_test[self.target].tolist()
        return X_train, y_train, X_test, y_test

    def _load_data(self, usecols=None, cols_only=False):
        """ Load dataframe from supplied path. Drop any rows with NaNs.
        Setting `labels_only` to `True` will load only the target variable.
        This is useful in cases where only that is needed, such as scoring.
        """
        nrows = None
        if cols_only:
            nrows = 0
        df = pd.read_table(self.filepath, usecols=usecols, nrows=nrows)
        self.data = df

    def _get_data_cols(self):
        """ Given a dataframe and its meta columns, get back a list of the data
        columns from the dataframe.
        """
        df = self.data
        cols = df.columns.tolist()
        data_cols = [x for x in cols if x not in self.meta_cols]
        self.data_cols = data_cols

    def transform_dataset(self, transform=None):
        if not transform:
            transforms = self.settings.get('transforms')
        else:
            transforms = [transform]
        if transforms:
            for transform in transforms:
                self._apply_transform(transform)

    def _apply_transform(self, transform):
        operation = transform.get("operation")
        if operation == "dropna":
            self.data = self.data.dropna(axis=0, how='any')
            self.meta_cols = [x for x in self.meta_cols
                              if x in self.data.columns.tolist()]
            self._get_data_cols()
        if operation == "drop_cols":
            columns_list = transform["columns_list"]
            self.data = self.data[[x for x in self.data.columns
                                   if x not in columns_list]]
            self.meta_cols = [x for x in self.meta_cols
                              if x in self.data.columns.tolist()]
            self._get_data_cols()
        if operation == "filter":
            column = transform["column"]
            values = transform["values"]
            self.data = self.data[self.data[column].isin(values)]
        if operation == "digitize":
            column = transform["column"]
            thresholds = transform["thresholds"]
            values = self.data[column].tolist()
            values = list(np.digitize(values, thresholds) - 1)
            self.data[column] = values
        if operation == "str2float":
            columns_list = transform["columns_list"]
            for column in columns_list:
                values = self.data[column].tolist()
                non_decimal = re.compile(r'[^\d.]+')
                converted_values = [non_decimal.sub('', x) for x in values]
                self.data[column] = converted_values
                self.data = self.data.astype({column: float})


def get_train_test_df(df, test_ids, column):
    """ Split a dataframe into a train/test portion
    given a column, and the list of testing set sample ids
    """
    df_train = df[~df[column].isin(test_ids)]
    df_test = df[df[column].isin(test_ids)]
    df_test.index = pd.Categorical(df_test[column], categories=test_ids)
    df_test = df_test.sort_index().reset_index(drop=True)
    return df_train, df_test
