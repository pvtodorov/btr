import csv
import argparse
import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestRegressor
import uuid
from tqdm import tqdm
import argparse
import json
import os
from statsmodels.sandbox.stats.multicomp import multipletests


def get_data_cols(df, meta_cols):
    """ Given a dataframe and its meta columns, get back a list of the data
    columns from the dataframe.
    """
    cols = df.columns.tolist()
    data_cols = [x for x in cols if x not in meta_cols]
    return data_cols


def get_gene_list_intersect(gene_list, data_cols):
    """ return the intersection between the current gene list HGNC symbols and
    the columns in the dataset. return a second list, `missing` for any genes
    that are missing.
    """
    intersect = [x for x in gene_list if x in data_cols]
    missing = [x for x in gene_list if x not in data_cols]
    if len(missing) > 0:
        print('Missing ' + str(len(missing)) + ' genes.')
        print(str(missing))
    return intersect, missing


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


def standardize_gmt(gmt):
    if 'http' in gmt[0][1]:
        gmt_standard = [[x[1]]+[x[0]]+x[2:] for x in gmt]
    else:
        gmt_standard = gmt
    return gmt_standard


def read_gmt(fpath):
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


def get_gmt_dict(gmt):
    d = {x[0]: {'meta': x[1], 'genes': x[2:]} for x in gmt}
    return d


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("settings_path", help="settings and parameters json")
    parser.add_argument("gmt_path", help="path to gmt file")
    args = parser.parse_args()
    settings_path = args.settings_path
    gmt_path = args.gmt_path
    with open(settings_path) as f:
        settings = json.load(f)

    infile = settings['infile']
    outfolder = settings['outfolder']
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    df = pd.read_table(infile)
    data_cols = get_data_cols(df, settings['meta_cols'])

    target = settings['target']
    subset = settings['subset']

    # make sure df target variable doesn't have NaNs
    df = df[(df[target].notnull())]
    # select a particular subset of the dataset based on BrodmannArea
    df = df[df['BrodmannArea'] == subset]
    # drop duplicate IDs, taking only the first time the ID occurs
    df = df.drop_duplicates(subset='ID', keep='first')
    data_cols = get_data_cols(df, settings['meta_cols'])
    print(df.shape)

    gmt = read_gmt(gmt_path)
    gmt_dict = get_gmt_dict(gmt)

    background = pd.read_csv(settings['outfolder'] + '.csv')
    bcg_cols = [int(x) for x in background.columns.tolist()]
    
    scores = []
    for g in tqdm(gmt):
        gene_list = g[2:]
        intersect, missing = get_gene_list_intersect(gene_list, data_cols)
        if len(intersect) < 1:
            print('No intersecting genes!')
            continue
        X, y = get_X_y(df, target, intersect)
        rf = fit_RF(X, y)
        s = {}
        s['id'] = g[0]
        s['description'] = g[1]
        s['R2'] = rf.oob_score_
        n_genes = len(intersect)
        b_idx = (np.abs(np.array(bcg_cols) - n_genes)).argmin()
        b_col = str(bcg_cols[b_idx])
        bcg_vals = background[b_col].tolist()
        bcg_vals_t = [x for x in bcg_vals if x > s['R2']]
        s['p_value'] = len(bcg_vals_t) / len(bcg_vals)
        scores.append(s)

    df_scores = pd.DataFrame(scores)

    if len(df_scores) > 1:
        p_values = df_scores['p_value'].tolist()
        mt = multipletests(p_values, alpha=0.05, method='fdr_bh')
        df_scores['adjusted_p'] = mt[1]
        df_scores = df_scores.sort_values(by=['adjusted_p', 'R2'],
                                          ascending=[True, False])

    gmt_suffix = gmt_path.split('/')[-1][:-4]
    df_scores.to_csv('scores_' + gmt_suffix + '.csv', index=False)


