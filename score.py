import pandas as pd
from tqdm import tqdm
import argparse
import json
import os
from utilities import (recursivedict, check_or_create_dir, get_outdir_path,
                       get_outfile_name, digitize_labels)
from dataset import Dataset
from gmt import GMT
import numpy as np
from pathlib import Path


class Scorer(object):
    def __init__(self, settings=None):
        self.s = settings
        self.y = []

    def from_settings(self, settings_path=None):
        if settings_path:
            with open(settings_path) as f:
                settings = json.load(f)
            self.s = settings
        if self.s:
            self.get_y_dict()

    def get_y_dict(self):
        """
        """
        dataset = Dataset(self.s)
        df = dataset.data
        id_col = dataset.id_col
        target = dataset.target
        ids = df[id_col].tolist()
        y = df[target].tolist()
        transform = self.s["processing_scheme"].get("transform_labels")
        y = digitize_labels(y, transform)
        y = [int(val) for val in y]
        y_dict = {}
        for i, y in zip(ids, y):
            y_dict[i] = y
        self.y = y_dict

    def score_LPOCV(self, gmt_path=None):
        outfolder = get_outdir_path(self.s, gmt=None)
        if gmt_path:
            outfolder = outfolder.split('background_predictions/')[0] + 'geneset_predictions/'
            gmt = GMT(gmt_path)
            bg_runs = [gmt.suffix + '.csv']
        else:
            bg_runs = os.listdir(outfolder)
            bg_runs = [x for x in bg_runs if '.csv' in x]
        aggregate_runs = pd.DataFrame()
        auc_dict_list = []
        for fn in tqdm(bg_runs):
            df = pd.read_csv(outfolder + fn)
            pair_idx = df['pair_index'].unique().tolist()
            col_numbers = [x for x in df.columns.tolist()
                           if x not in ['ID', 'pair_index']]
            pairs = recursivedict()
            for p_idx in pair_idx:
                df_0 = df[df['pair_index'] == p_idx]
                y_true = [self.y[x] for x in df_0['ID'].tolist()]
                diff_true = np.diff(y_true)[0]
                sign_true = np.sign(diff_true)
                for col_n in col_numbers:
                    y_pred = df_0[col_n].tolist()
                    diff_pred = np.diff(y_pred)[0]
                    sign_pred = np.sign(diff_pred)
                    diff_sign = np.sign(np.diff([sign_true, sign_pred])[0])
                    if diff_pred == 0:
                        pairs[col_n][p_idx] = 0.5
                    else:
                        if diff_sign == 0:
                            pairs[col_n][p_idx] = 1
                        else:
                            pairs[col_n][p_idx] = 0
            out = pd.DataFrame(pairs)
            cols = out.columns.tolist()
            if not gmt_path:
                out = out.rename(columns={a: int(a) for a in cols})
            cols = out.columns.tolist()
            out = out[list(sorted(cols))]
            out = dict(out.mean())
            auc_dict_list.append(out)
        auc_df = pd.DataFrame(auc_dict_list)
        cols = auc_df.columns.tolist()
        if not gmt_path:
            auc_df = auc_df.rename(columns={a: int(a) for a in cols})
        cols = auc_df.columns.tolist()
        auc_df = auc_df[list(sorted(cols))]
        if 'background_predictions/' in outfolder:
            outfolder = outfolder.split('background_predictions/')[0]
        if 'geneset_predictions/' in outfolder:
            outfolder = outfolder.split('geneset_predictions/')[0]
        if gmt_path:
            auc_df.to_csv(outfolder + gmt.suffix + '_auc.csv', index=False)
        else:
            auc_df.to_csv(outfolder + 'background_auc.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("settings_path", help="settings as JSON")
    parser.add_argument("-g", "--gmt_path",
                        help="path to file or folder of txts", required=False)
    argument = parser.parse_args()
    args = parser.parse_args()
    settings_path = args.settings_path
    gmt_path = args.gmt_path
    gmt = None
    if gmt_path:
        gmt = GMT(gmt_path)
    scorer = Scorer()
    scorer.from_settings(settings_path)
    scorer.score_LPOCV(gmt_path=gmt_path)