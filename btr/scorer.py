import os

import numpy as np
import pandas as pd
from statsmodels.sandbox.stats.multicomp import multipletests
from tqdm import tqdm

from .processing_schemes import Processor
from .utilities import recursivedict, get_outdir_path


class Scorer(Processor):
    def __init__(self, settings=None):
        super().__init__(settings=settings)
        self.annotations['type'] = 'Scorer'
        self.df = pd.DataFrame()
        self.y_dict = {}

    def get_y_dict(self, dataset):
        """
        """
        ids = dataset.data[dataset.id_col].tolist()
        y = dataset.data[dataset.target].tolist()
        y = [int(val) for val in y]
        y_dict = {}
        for i, y in zip(ids, y):
            y_dict[i] = y
        self.y_dict = y_dict

    def get_score(self, gmt=None):
        pass


class ScoreLPOCV(Scorer):
    def __init__(self, settings=None):
        super().__init__(settings=settings)
        pass

    def get_score(self, gmt=None):
        infolder = get_outdir_path(self.settings)
        if gmt:
            self.annotations['gmt'] = gmt.suffix
            infolder += 'hypothesis_predictions/'
            file_names = [gmt.suffix + '.csv']
        else:
            infolder += 'background_predictions/'
            file_names = os.listdir(infolder)
            file_names = [x for x in file_names if '.csv' in x]
        auc_dict_list = []
        for fn in tqdm(file_names):
            df = pd.read_csv(infolder + fn)
            pairs = get_pair_auc_dict(df, self.y_dict)
            out = pd.DataFrame(pairs)
            cols = out.columns.tolist()
            if not self.annotations.get('gmt'):
                out = out.rename(columns={a: int(a) for a in cols})
            cols = out.columns.tolist()
            out = out[list(sorted(cols))]
            out = dict(out.mean())
            auc_dict_list.append(out)
        auc_df = pd.DataFrame(auc_dict_list)
        cols = auc_df.columns.tolist()
        if not self.annotations.get('gmt'):
            auc_df = auc_df.rename(columns={a: int(a) for a in cols})
        cols = auc_df.columns.tolist()
        auc_df = auc_df[list(sorted(cols))]
        self.annotations['score_metric'] = 'AUC'
        if self.annotations.get('gmt'):
            self.annotations['score_type'] = 'hypothesis'
            self.annotations['gmt'] = gmt.suffix
        else:
            self.annotations['score_type'] = 'hypothesis'
        self.df = auc_df

    def get_stats(self, gmt=None, dataset=None):
        folder = get_outdir_path(self.settings) + 'score/'
        scored_predictions = pd.read_csv(folder + gmt.suffix + '_auc.csv')
        background = pd.read_csv(folder + 'background_auc.csv')
        bcg_cols = background.columns.tolist()
        bcg_cols = [int(x) for x in bcg_cols]
        d_cols = dataset.data_cols
        scores = []
        for link, desc, g_list, m_list in gmt.generate(dataset_genes=d_cols):
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
        folder = "/".join(folder.split('/')[:-2] + ['stats', ''])
        filepath = folder + gmt.suffix + '_stats.csv'
        self.df = df_scores
        self.annotations['stats_metric'] = 'AUC'
        if self.annotations.get('gmt'):
            self.annotations['stats_type'] = 'hypothesis'
            self.annotations['gmt'] = gmt.suffix
        else:
            self.annotations['stats_type'] = 'background'


def get_pair_auc_dict(df, y_dict):
    pair_auc_dict = recursivedict()
    predict_meta_cols = ['pair_index', 'ID', 'class']
    predict_data_cols = [x for x in df.columns.tolist()
                         if x not in predict_meta_cols]
    for col in predict_data_cols:
        cols_f = predict_meta_cols + [col]
        df_t = df.loc[:, :].copy()
        df_t = df_t.loc[:, cols_f]
        df_t.loc[:, 'true'] = [y_dict[x] for x in df_t.loc[:, 'ID'].tolist()]
        sort_cols = ['pair_index', 'true', col, 'class']
        cols_ascending = [True, True, False, True]
        df_t.sort_values(sort_cols,
                         ascending=cols_ascending,
                         inplace=True)
        df_t.drop_duplicates(subset=['ID', 'pair_index'],
                             keep='first',
                             inplace=True)
        pair_idx_list = list(set(df_t['pair_index'].tolist()))
        for pair_idx in pair_idx_list:
            df_p = df_t.loc[df_t['pair_index'] == pair_idx, :]
            sample_class_list = df_p['class'].tolist()
            lo = sample_class_list[0]
            hi = sample_class_list[1]
            if lo == hi:
                probabilities_list = df_p[col].tolist()
                if lo == 0:
                    probabilities_list = list(reversed(probabilities_list))
                lo = probabilities_list[0]
                hi = probabilities_list[1]
            auc = calculate_pair_auc(lo, hi)
            pair_auc_dict[col][pair_idx] = auc
    return pair_auc_dict


def calculate_pair_auc(lo, hi):
    if lo == hi:
        auc = 0.5
    elif lo < hi:
        auc = 1
    else:
        auc = 0
    return auc
