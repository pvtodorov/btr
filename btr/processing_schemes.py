import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import combinations
from .utilities import (recursivedict, check_or_create_dir, get_outdir_path,
                        get_outfile_name, digitize_labels)


class Processor(object):
    """Processor base class.
    The purpose of processors is to allow a user to produce predictions of both
    background and gmt gene lists.
    """

    def __init__(self, settings=None, dataset=None, estimator=None):
        self.settings = settings
        self.dataset = dataset
        self.estimator = estimator


class LPOCV(Processor):
    """Leave-Pair-Out Cross-Validation scheme

    Implements leave-pair-out cross-validation in which each pairs of samples
    are chosen from the dataset followed by training on all but one pair and
    predicting the target variable for the withheld pair.
    """

    def __init__(self, settings=None, dataset=None, estimator=None):
        super().__init__(settings=settings,
                         dataset=dataset,
                         estimator=estimator)
        self.selected_pairs = []
        self.df_result = pd.DataFrame()
        self.gmt = None
        self._bcg_predictions = recursivedict()
        self._pairs_list = []

    def predict(self, gmt=None):
        """Performs a background or hypothesis prediction

        Reads in lists of features from a `gmt` or a random feature list for
        background and uses them to fit a model and make a predictions
        """
        if gmt:
            self.gmt = gmt
            data_cols = self.dataset.data_cols
            for link, _, gene_list, _ in tqdm(gmt.generate(data_cols),
                                              total=len(gmt.gmt)):
                self._build_bcg_predictions(gene_list, link)
        else:
            sampling_range = get_sampling_range(self.settings)
            for k in tqdm(sampling_range):
                gene_list = self.dataset.sample_data_cols(k)
                self._build_bcg_predictions(gene_list, k)
        self._build_df_result()

    def save_results(self):
        """Saves prediction results as csv files.

        Random gene set predictions are placed in `background_predictions/`
        Gene set predictions are place in `hypothesis_predictions/`
        """
        self._outdir_path = get_outdir_path(self.settings)
        if self.gmt:
            self._outdir_path += 'hypothesis_predictions/'
        else:
            self._outdir_path += 'background_predictions/'
        check_or_create_dir(self._outdir_path)
        self._outfile_name = get_outfile_name(gmt=self.gmt)
        results_path = self._outdir_path + self._outfile_name
        self.df_result.to_csv(results_path, index=False)

    def _build_bcg_predictions(self, selected_cols, k):
        """Performs LPOCV and updates the Processor's `_bcg_predictions` dict

        This function is given `selected_cols` corresponding to the selected
        features and `k` a value corresponding to what these selected features
        should be called. In the random case, `k` is a number of selected
        features. In the case where a feature set is supplied via `gmt`, `k`
        corresponds to `gmt.suffix`
        """
        self._get_pairs(**self.settings["processing_scheme"]["pair_settings"])
        for pair_index, pair in enumerate(tqdm(self.selected_pairs)):
                tf = self.settings["processing_scheme"].get("transform_labels")
                pair_ids = (pair[0][0], pair[1][0])
                dataset = self.dataset
                train_test_list = dataset.get_X_y(pair_ids,
                                                  selected_cols=selected_cols)
                X_train, y_train, X_test, _ = train_test_list
                y_train = [int(x) for x in y_train]
                y_train = np.array(y_train)
                y_train = digitize_labels(y_train, tf)
                e = self.estimator
                e = e.fit(X_train, y_train)
                if self.settings["estimator"].get("call") == "probability":
                    predictions = e.predict_proba(X_test)[:, 1]
                else:
                    predictions = e.predict(X_test)
                for i_s, p in zip(pair_ids, predictions):
                    self._bcg_predictions[pair_index][i_s][k] = p

    def _build_df_result(self):
        """Produces a dataframe from `self._bcg_predictions`"""
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
        """Generates all possible pairs of samples.

        From `settings`
        - Filters dataframe such that `subset_col` is limited to `subset`
        - uses `transform` to `digitize_labels`
        - creates all possible pairs of samples
        """
        dataset = self.dataset
        settings = self.settings
        tf = settings["processing_scheme"].get("transform_labels")
        subset_column = settings["processing_scheme"]["subset_col"]
        subset = settings["processing_scheme"]["subset"]
        df = dataset.data
        df_f = df[df[subset_column] == subset]
        ids_list = df_f[dataset.id_col].tolist()
        splits_list = df_f[dataset.target].tolist()
        if tf:
            splits_list = digitize_labels(splits_list, tf)
        samples = [(x, y) for x, y in
                   zip(ids_list, splits_list)]
        samples = sorted(samples, key=lambda x: x[1])
        pairs_list = [x for x in combinations(samples, 2)]
        self._pairs_list = pairs_list

    def _get_pairs(self, shuffle=True, seed=47, sample_once=False):
        """Generates reproducible list of pairs to use for LPOCV.

        Keyword arguments:
        shuffle -- whether or not to shuffle the pairs from
                   `_build_pairs_list()`. (default True)
        seed -- seed to use when shuffling pairs. (default 47)
        sample_once -- makes an effort to include each in the dataset at most
                       once. (default False)
        """
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
    """Returns a list of feature set lengths for background sampling.
    """
    interval_params_list = settings['background_params']['intervals']
    sampling_range = []
    for params in interval_params_list:
        start = params['start']
        end = params['end']
        step = params['step']
        range_section = list(range(start, end + step, step))
        sampling_range += range_section
    return sampling_range
