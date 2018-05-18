import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from statsmodels.sandbox.stats.multicomp import multipletests
from .utilities import (recursivedict, get_outdir_path, check_or_create_dir)
from .processing_schemes import Processor


class Scorer(Processor):
    def __init__(self, settings=None):
        super().__init__(settings=settings)
        self.annotations['type'] = 'Scorer'
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

    def get_intersect():
        pass


class ScoreLPOCV(Scorer):
    def __init__(self, settings=None):
        super().__init__(settings=settings)
        pass

    def score_LPOCV(self, gmt=None):
        outfolder = get_outdir_path(self.settings)
        if gmt:
            self.annotations['gmt'] = gmt.suffix
            outfolder += 'hypothesis_predictions/'
            check_or_create_dir(outfolder)
            bg_runs = [gmt.suffix + '.csv']
        else:
            outfolder += 'background_predictions/'
            check_or_create_dir(outfolder)
            bg_runs = os.listdir(outfolder)
            bg_runs = [x for x in bg_runs if '.csv' in x]
        auc_dict_list = []
        bg_runs = os.listdir(outfolder)
        for fn in tqdm(bg_runs):
            df = pd.read_csv(outfolder + fn)
            pair_idx = df['pair_index'].unique().tolist()
            col_numbers = [x for x in df.columns.tolist()
                           if x not in ['ID', 'pair_index']]
            pairs = recursivedict()
            for p_idx in pair_idx:
                df_0 = df[df['pair_index'] == p_idx]
                y_true = [self.y_dict[x] for x in df_0['ID'].tolist()]
                for col_n in col_numbers:
                    y_pred = df_0[col_n].tolist()
                    y_pred_sorted = [x for _, x in
                                     sorted(zip(y_true, y_pred),
                                            key=lambda y: y[0])]
                    lo, hi = y_pred_sorted[0], y_pred_sorted[1]
                    if lo == hi:
                        auc = 0.5
                    elif lo < hi:
                        auc = 1
                    else:
                        auc = 0
                    pairs[col_n][p_idx] = auc
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
        outfolder = "/".join(outfolder.split('/')[:-2] + ['score', ''])
        check_or_create_dir(outfolder)
        self.annotations['btr_file_type'] = 'score'
        self.annotations['score_metric'] = 'AUC'
        if self.annotations.get('gmt'):
            filepath = outfolder + gmt.suffix + '_auc.csv'
            self.annotations['score_type'] = 'hypothesis'
            self.annotations['gmt'] = gmt.suffix
        else:
            filepath = outfolder + 'background_auc.csv'
            self.annotations['score_type'] = 'hypothesis'
        auc_df.to_csv(filepath, index=False)

    def get_stats(self, gmt=None, dataset=None):
        folder = get_outdir_path(self.settings) + 'score/'
        check_or_create_dir(folder)
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
        check_or_create_dir(folder)
        filepath = folder + gmt.suffix + '_stats.csv'
        df_scores.to_csv(filepath, index=False)
        self.annotations['btr_file_type'] = 'stats'
        self.annotations['score_metric'] = 'AUC'
        self.annotations['gmt'] = gmt.suffix
