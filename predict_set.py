import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os
from statsmodels.sandbox.stats.multicomp import multipletests
from msbb_functions import *


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

    gmt = read_gmt(gmt_path)
    background = pd.read_csv(settings['outfolder'] + '.csv')
    bcg_cols = [int(x) for x in background.columns.tolist()]

    scores = []
    for g in tqdm(gmt):
        gene_list = g[2:]
        intersect, missing = get_gene_list_intersect(gene_list, data_cols)
        if len(intersect) < 1:
            continue
        X, y = get_X_y(df, target, intersect)
        rf = fit_RF(X, y)
        s = {}
        s['id'] = g[0]
        s['description'] = g[1]
        s['R2'] = rf.oob_score_
        s['n_genes'] = len(gene_list)
        s['intersect'] = len(intersect)
        b_idx = (np.abs(np.array(bcg_cols) - len(intersect))).argmin()
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
    df_scores = df_scores[['id', 'description', 'n_genes', 'intersect',
                           'R2', 'p_value', 'adjusted_p']]
    df_scores.to_csv('scores_' + gmt_suffix + '.csv', index=False)
