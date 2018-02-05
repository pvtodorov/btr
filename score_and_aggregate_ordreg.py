import pandas as pd
from tqdm import tqdm
import argparse
import json
import os
from msbb_functions import *


def get_y_dict(df, target):
    """ 
    """
    ids = df['ID'].tolist()
    y = df[target].tolist()
    y = [int(val) for val in y]
    y_dict = {}
    for i, y in zip(ids, y):
        y_dict[i] = y
    return y_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("settings_path", help="settings and parameters json")
    args = parser.parse_args()
    settings_path = args.settings_path
    with open(settings_path) as f:
        settings = json.load(f)

    infile = settings['infile']
    outfolder = settings['outfolder']
    target = settings['target']
    subset_col = settings['subset_col']
    subset = settings['subset']
    pair_col = settings['pair_col']
    interval = settings['interval']
    max_cols = settings['max_cols']

    data = load_msbb_data(infile)
    data = data[data[subset_col] == subset]
    data_cols = get_data_cols(data, settings['meta_cols'])

    y_dict = get_y_dict(data, target)
    y_dict_digitized = {i: digitize_Braak_scores(y) for i, y in y_dict.items()}

    outfolder = settings['outfolder'] + '/' + settings['subset']
    bg_runs = os.listdir(outfolder)
    bg_runs = [x for x in bg_runs if '.csv' in x]
    aggregate_runs = pd.DataFrame()
    auc_dict_list = []
    for fn in tqdm(bg_runs):
        df = pd.read_csv(outfolder + '/' + fn)
        pair_idx = df['pair_index'].unique().tolist()
        col_numbers = [x for x in df.columns.tolist() if x not in ['ID', 'pair_index']]
        pairs = recursivedict()
        for p_idx in pair_idx:
            df_0 = df[df['pair_index'] == p_idx]
            y_true = [y_dict_digitized[x] for x in df_0['ID'].tolist()]
            diff_true = np.diff(y_true)[0]
            sign_true = np.sign(diff_true)
            for col_n in col_numbers:
                y_pred = df_0[col_n].tolist()
                diff_pred = np.diff(y_pred)[0]
                sign_pred = np.sign(diff_pred)
                diff_sign = np.sign(np.diff([sign_true, sign_pred])[0])
                if diff_sign == 0:
                    if diff_true == diff_pred:
                        pairs[col_n][p_idx] = 1
                    else:
                        pairs[col_n][p_idx] = 0.5
                else:
                    pairs[col_n][p_idx] = 0
        out = pd.DataFrame(pairs)
        cols = out.columns.tolist()
        out = out.rename(columns={a:int(a) for a in cols})
        cols = out.columns.tolist()
        out = out[list(sorted(cols))]
        out = dict(out.mean())
        auc_dict_list.append(out)
    auc_df = pd.DataFrame(auc_dict_list)
    cols = auc_df.columns.tolist()
    auc_df = auc_df.rename(columns={a:int(a) for a in cols})
    cols = auc_df.columns.tolist()
    auc_df = auc_df[list(sorted(cols))]
    auc_df.to_csv(outfolder + '.csv', index=False)
