import pandas as pd
import numpy as np


class Dataset(object):
    def __init__(self, settings, usecols=None,
                 filter_dataset=True, transform_dataset=True):
        self.settings = settings
        self.name = self.settings['dataset']['name']
        self.filepath = self.settings['dataset']['filepath']
        self.meta_cols = self.settings['dataset']['meta_columns']
        self.target = self.settings['dataset']['target']
        self.id_col = self.settings['dataset']['ID_column']
        self.data_cols = []
        self.data = pd.DataFrame()
        self._load_data(usecols=usecols)
        self._get_data_cols()
        if filter_dataset:
            self.filter_dataset()
        if transform_dataset:
            self.transform_dataset()

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
        X_train = df_train.as_matrix(columns=selected_cols)
        y_train = df_train[self.target].tolist()
        X_test = df_test.as_matrix(columns=selected_cols)
        y_test = df_test[self.target].tolist()
        return X_train, y_train, X_test, y_test

    def _load_data(self, usecols=None):
        """ Load dataframe from supplied path. Drop any rows with NaNs.
        Setting `labels_only` to `True` will load only the target variable.
        This is useful in cases where only that is needed, such as scoring.
        """
        df = pd.read_table(self.filepath, usecols=usecols)
        df = df.dropna(axis=0, how='any')
        self.data = df

    def _get_data_cols(self):
        """ Given a dataframe and its meta columns, get back a list of the data
        columns from the dataframe.
        """
        df = self.data
        cols = df.columns.tolist()
        data_cols = [x for x in cols if x not in self.meta_cols]
        self.data_cols = data_cols

    def filter_dataset(self, filters=None):
        """ A filter defined in settings['dataset']['filter']['filters'] can
        be applied to the loaded dataset. An example filter:
        "filter": {"name": "AB",
                   "filters": [{"column": "Braak", "values": [0, 1, 2, 3, 4]}]
                  }

        The filter "name" is used in naming the folder that output files are
        deposited into. The "filters" list specifies a list of dicts, each
        of which defines a "column" to limit and a list of values to limit the
        column to.
        """
        if not filters:
            if self.settings['dataset'].get('filter'):
                filters = (self.settings['dataset']['filter']['filters'])
        data = self.data
        if filters:
            for f in filters:
                data = data[data[f['column']].isin(f['values'])]
        self.data = data

    def transform_dataset(self, transform=None):
        if not transform:
            transform = self.settings['dataset'].get('transform')
        if transform:
            if transform.get('digitize'):
                digitize_param = transform.get('digitize')
                for tf in digitize_param:
                    column = tf['column']
                    tresholds = tf['thresholds']
                    values = self.data[column].tolist()
                    values = list(np.digitize(values, tresholds) - 1)
                    self.data[column] = values


def get_train_test_df(df, test_ids, column):
    """ Split a dataframe into a train/test portion
    given a column, and the list of testing set sample ids
    """
    df_train = df[~df[column].isin(test_ids)]
    df_test = df[df[column].isin(test_ids)]
    df_test.index = pd.Categorical(df_test[column], categories=test_ids)
    df_test = df_test.sort_index().reset_index(drop=True)
    return df_train, df_test
