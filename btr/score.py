import pandas as pd
from tqdm import tqdm
import argparse
import json
import os
from .utilities import (recursivedict, get_outdir_path, digitize_labels)
from .dataset import Dataset
from .gmt import GMT
import numpy as np
from statsmodels.sandbox.stats.multicomp import multipletests


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
        outfolder = get_outdir_path(self.s)
        if gmt_path:
            outfolder += 'geneset_predictions/'
            gmt = GMT(gmt_path)
            bg_runs = [gmt.suffix + '.csv']
        else:
            outfolder += 'background_predictions/'
            bg_runs = os.listdir(outfolder)
            bg_runs = [x for x in bg_runs if '.csv' in x]
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
        if gmt_path:
            auc_df.to_csv(outfolder + '../' + gmt.suffix + '_auc.csv', index=False)
        else:
            auc_df.to_csv(outfolder + '../' + 'background_auc.csv', index=False)

    def get_stats(self, gmt_path):
        folder = get_outdir_path(self.s)
        gmt = GMT(gmt_path)
        dataset = Dataset(self.s)
        scored_predictions = pd.read_csv(folder + gmt.suffix + '_auc.csv')
        background = pd.read_csv(folder + 'background_auc.csv')
        bcg_cols = background.columns.tolist()
        bcg_cols = [int(x) for x in bcg_cols]

        data_cols = dataset.data_cols
        scores = []
        for link, desc, g_list, m_list in gmt.generate(dataset_genes=data_cols):
            gene_list = g_list + m_list
            intersect = g_list
            if len(intersect) < 1:
                continue
            s = {}
            s['id'] = link
            s['description'] = desc
            s['AUC'] = scored_predictions[link].tolist()[0]
            s['n_genes'] = len(gene_list)
            s['intersect'] = len(intersect)
            b_idx = (np.abs(np.array(bcg_cols) - len(intersect))).argmin()
            b_col = str(bcg_cols[b_idx])
            bcg_vals = background[b_col].tolist()
            bcg_vals_t = [x for x in bcg_vals if x > s['AUC']]
            s['p_value'] = len(bcg_vals_t) / len(bcg_vals)
            scores.append(s)

        df_scores = pd.DataFrame(scores)

        p_values = df_scores['p_value'].tolist()
        mt = multipletests(p_values, alpha=0.05, method='fdr_bh')
        df_scores['adjusted_p'] = mt[1]
        df_scores = df_scores.sort_values(by=['adjusted_p', 'AUC'],
                                          ascending=[True, False])

        df_scores = df_scores[['id', 'description', 'n_genes', 'intersect',
                               'AUC', 'p_value', 'adjusted_p']]
        df_scores.to_csv(folder + gmt.suffix + '_stats.csv', index=False)


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
