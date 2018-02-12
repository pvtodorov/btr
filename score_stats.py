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
from statsmodels.sandbox.stats.multicomp import multipletests


class Loader(object):
    def __init__(self, settings=None):
        self.s = settings
        self.y = []

    def from_settings(self, settings_path=None):
        if settings_path:
            with open(settings_path) as f:
                settings = json.load(f)
            self.s = settings

    def get_stats(self, gmt_path):
        folder = (self.s['dataset']['name'] + '/' +
                  self.s["processing_scheme"]["name"] + '/' +
                  self.s['processing_scheme']['subset'] + '/' +
                  self.s["estimator"]["name"] + '/')
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
    scorer = Loader()
    scorer.from_settings(settings_path)
    scorer.get_stats(gmt_path=gmt_path)