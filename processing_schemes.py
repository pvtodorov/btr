import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import combinations
from msbb_functions import *
import uuid
import json
from dataset import Dataset
from estimators import get_estimator


class Loader(object):
    @staticmethod
    def processor_from_settings(settings_path):
        if settings_path:
            with open(settings_path) as f:
                settings = json.load(f)
        name = settings["processing_scheme"]["name"]
        if name == 'LPOCV':
            return LPOCV(settings=settings)
        else:
            raise NotImplementedError


class Processor(object):
    def __init__(self, settings=None, dataset=None, estimator=None):
        self.s = None
        if settings:
            self.s = settings
        self.d = None
        if dataset:
            self.d = dataset
        self.e = None
        if estimator:
            self.e = estimator

    def from_settings(self, settings_path=None):
        if settings_path:
            with open(settings_path) as f:
                settings = json.load(f)
            self.s = settings
        if self.s:
            self.d = Dataset(self.s)
            self.e = get_estimator(self.s)


class LPOCV(Processor):
    def __init__(self, settings=None, dataset=None, estimator=None):
        super().__init__(settings=settings,
                         dataset=dataset,
                         estimator=estimator)
        self.selected_pairs = []
        self.df_result = pd.DataFrame()
        self._bcg_predictions = recursivedict()
        self._pairs_list = []
        self._transform = settings["processing_scheme"].get("transform_labels")
        self._outdir_path = ''
        self._outfile_name = ''

    def predict_background(self):
        sampling_range = get_sampling_range(self.s)
        for k in tqdm(sampling_range):
            gene_list = self.d.sample_data_cols(k)
            self._build_bcg_predictions(gene_list, k)
        self._build_df_result()

    def predict_gmt(self, gmt):
        for link, _, gene_list, _ in tqdm(gmt.generate(self.d.data_cols)):
            self._build_bcg_predictions(gene_list, link)
        self._build_df_result()

    def save_results(self, gmt=None):
        self._get_outdir_path(gmt)
        check_or_create_dir(self._outdir_path)
        self._get_outfile_name(gmt)
        results_path = self._outdir_path + self._outfile_name
        self.df_result.to_csv(results_path, index=False)

    def _get_outdir_path(self, gmt):
        dset_name = self.s["dataset"]["name"]
        scheme_name = self.s["processing_scheme"]["name"]
        subset_name = self.s["processing_scheme"]["subset"]
        est_name = self.s["estimator"]["name"]
        prediction_type = "background_predictions"
        if gmt:
            prediction_type = "geneset_predictions"
        outdir_path = (dset_name + '/' +
                       scheme_name + '/' + subset_name + '/' +
                       est_name + '/' +
                       prediction_type + '/')
        self._outdir_path = outdir_path

    def _get_outfile_name(self, gmt):
        if gmt:
            outfile_name = gmt.suffix
        else:
            outfile_name = str(uuid.uuid4())
        outfile_name = outfile_name + '.csv'
        self._outfile_name = outfile_name

    def _build_bcg_predictions(self, selected_cols, k):
        self._get_pairs(**self.s["processing_scheme"]["pair_settings"])
        for pair_index, pair in enumerate(tqdm(self.selected_pairs)):
                pair_ids = (pair[0][0], pair[1][0])
                train_test_list = self.d.get_X_y(pair_ids,
                                                 selected_cols=selected_cols)
                X_train, y_train, X_test, _ = train_test_list
                y_train = [int(x) for x in y_train]
                y_train = np.array(y_train)
                y_train = digitize_labels(y_train, self._transform)
                e = self.e
                e = e.fit(X_train, y_train)
                predictions = e.predict(X_test)
                for i_s, p in zip(pair_ids, predictions):
                    self._bcg_predictions[pair_index][i_s][k] = p

    def _build_df_result(self):
        for pair_index, p in self._bcg_predictions.items():
            df_result_t = pd.DataFrame(p)
            df_result_t = df_result_t.transpose()
            df_result_t = df_result_t.reset_index()
            df_result_t = df_result_t.rename(columns={'index': 'ID'})
            df_result_t['pair_index'] = pair_index
            df_result_t_cols = df_result_t.columns.tolist()
            df_result_t_cols = sorted([x for x in df_result_t_cols
                                       if x not in ['ID', 'pair_index']])
            df_result_t_cols = ['ID', 'pair_index'] + df_result_t_cols
            df_result_t = df_result_t[df_result_t_cols]
            self.df_result = self.df_result.append(df_result_t)

    def _build_pairs_list(self):
        """ generate pairs of samples """
        dataset = self.d
        settings = self.s
        transform = self._transform
        subset_column = settings["processing_scheme"]["subset_col"]
        subset = settings["processing_scheme"]["subset"]
        df = dataset.data
        df_f = df[df[subset_column] == subset]
        ids_list = df_f[dataset.id_col].tolist()
        splits_list = df_f[dataset.target].tolist()
        if transform:
            splits_list = digitize_labels(splits_list, transform)
        samples = [(x, y) for x, y in
                   zip(ids_list, splits_list)]
        samples = sorted(samples, key=lambda x: x[1])
        pairs_list = [x for x in combinations(samples, 2)]
        self._pairs_list = pairs_list

    def _get_pairs(self, shuffle=True, seed=47, sample_once=False):
        """ Added function to return a set of pairs that have each sample
        in data at least once """
        self.selected_pairs = []
        if len(self._pairs_list) == 0:
            self._build_pairs_list()
        sample_list = sorted(list(set([sample[0] for pair
                                       in self._pairs_list
                                       for sample in pair])))
        used_ids = []
        selected_pairs = []
        prng = np.random.RandomState(seed)
        for sample in sample_list:
            if sample_once:
                if sample in used_ids:
                    continue
            pairs_list_f0 = [x for x in self._pairs_list]
            pairs_list_f1 = [x for x in pairs_list_f0
                             if sample in [x[0][0], x[1][0]]]
            pairs_list_f2 = [x for x in pairs_list_f1
                             if x[0][1] != x[1][1]]
            pairs_list_f3 = [x for x in pairs_list_f2
                             if x not in selected_pairs]
            pairs_list_f4 = [x for x in pairs_list_f3
                             if ((x[0][0] not in used_ids) and
                                 (x[1][0] not in used_ids))]
            if len(pairs_list_f4) > 0:
                pairs_list_f3 = pairs_list_f4
            sel_pair = pairs_list_f3[prng.choice(len(pairs_list_f3)) - 1]
            selected_ids = [sample[0] for sample in sel_pair]
            used_ids += selected_ids
            self.selected_pairs.append(sel_pair)


def get_sampling_range(settings):
    interval_params_list = settings['background_params']['intervals']
    sampling_range = []
    for params in interval_params_list:
        start = params['start']
        end = params['end']
        step = params['step']
        range_section = list(range(start, end + step, step))
        sampling_range += range_section
    return sampling_range


def digitize_labels(y, transform_thresholds):
    """ Take Braak scores and digitize them such that:
    [0, 1, 2, 3] -> 0
    [4, 5, 6]    -> 1
    """
    y_bin = np.digitize(y, transform_thresholds) - 1
    return y_bin
