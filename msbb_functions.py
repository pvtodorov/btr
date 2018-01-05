import pandas as pd
import random
from sklearn.ensemble import RandomForestRegressor
import uuid
from tqdm import tqdm
import csv


def get_data_cols(df, meta_cols):
    """ Given a dataframe and its meta columns, get back a list of the data
    columns from the dataframe.
    """
    cols = df.columns.tolist()
    data_cols = [x for x in cols if x not in meta_cols]
    return data_cols


def sample_data_cols(data_cols, k):
    """ Select a random sample of k-items from a list of columns.
    """
    sampled_cols = random.sample(data_cols, k)
    return [x for x in data_cols if x in sampled_cols]


def get_X_y(df, target, data_cols):
    """ Given a dataframe, a target col, and a sample of data column names,
    generate X, an numpy array of the data, and y, a list of target values
    corresponding to each row.
    """
    X = df.as_matrix(columns=data_cols)
    y = df[target].tolist()
    return X, y


def fit_RF(X, y):
    """ Fit and return a randomForestRegressor object to the provided data """
    rf = RandomForestRegressor(n_estimators=100,
                               max_features='auto',
                               n_jobs=4,
                               oob_score=True)
    rf.fit(X, y)
    return rf


def gen_background_performance(df, target, data_cols,
                               interval=10,
                               max_cols=1000):
    """ Given a dataframe, a target variable, data columns, a max number of
    columns to fit, and an interval to increase the sample of columns. Fits
    the data over the intervals and returns the oob R2 scores for each fit
    as a dict of {n_features: R2, ...}
    """
    oob_scores = {}
    for k in tqdm(range(10, max_cols + interval, interval)):
        selected_cols = sample_data_cols(data_cols, k)
        X, y = get_X_y(df, target, selected_cols)
        rf = fit_RF(X, y)
        oob_scores[k] = rf.oob_score_
    return oob_scores


def save_oob_scores(oob_scores, folder):
    """ Given a dict of oob_scores and a folder path, dump a csv of the
    oob_scores into the folder """
    d = oob_scores
    scores = pd.DataFrame(list(d.items()), columns=['n_features', 'R2'])
    scores = scores.sort_values('n_features').reset_index(drop=True)
    outfile = str(uuid.uuid4())
    scores.to_csv(folder + '/' + outfile + '.csv', index=False)


def get_gene_list_intersect(gene_list, data_cols):
    """ return the intersection between the current gene list HGNC symbols and
    the columns in the dataset. return a second list, `missing` for any genes
    that are missing.
    """
    intersect = [x for x in gene_list if x in data_cols]
    missing = [x for x in gene_list if x not in data_cols]
    return intersect, missing


def standardize_gmt(gmt):
    """ Takes a loaded list from a .gmt file and reformats it, if necessary,
    so that the html id is always at index 0 and the description is at index 1
    """
    if 'http' in gmt[0][1]:
        gmt_standard = [[x[1]] + [x[0]] + x[2:] for x in gmt]
    else:
        gmt_standard = gmt
    return gmt_standard


def read_gmt(fpath):
    """ given a filepath, reads the gmt or txt file at that location, returning
    a list that can be used in the scripts
    """
    if '.gmt' == fpath[-4:]:
        gmt = []
        with open(fpath) as f:
            rd = csv.reader(f, delimiter="\t", quotechar='"')
            for row in rd:
                gmt.append(row)
        gmt = standardize_gmt(gmt)
        # gmt = gmt[:100]
        # GET RID OF THIS
        # LIMIT TO 100 LINES FOR TESTING ONLY!!!!
        return gmt
    elif '.txt' == fpath[-4:]:
        with open(fpath) as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            gene_list = []
            for row in rd:
                gene_list.append(row[0])
            gmt = [[fpath.split('/')[-1]] + ['user defined'] + gene_list]
        return gmt
