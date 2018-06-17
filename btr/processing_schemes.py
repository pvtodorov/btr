import numpy as np
import pandas as pd
from tqdm import tqdm

from .estimators import get_estimator
from .pairs_processor import PairsProcessor
from .utilities import (check_or_create_dir, get_outdir_path, get_uuid,
                        recursivedict, save_json)


class Processor(object):
    """Processor base class.
    The purpose of processors is to allow a user to produce predictions of both
    background and gmt gene lists.
    """

    def __init__(self, settings=None, dataset=None):
        self.settings = settings
        self.annotations = {}


class Predictor(Processor):
    def __init__(self, settings=None, dataset=None):
        super().__init__(settings=settings, dataset=dataset)
        self.dataset = dataset
        self.annotations['type'] = 'Predictor'
        self.estimator = None
        if self.settings:
            self.get_estimator(self.settings.get('estimator'))
        self.uuid = get_uuid()
        self.annotations['uuid'] = str(self.uuid)
        self.df_result = pd.DataFrame()

    def get_estimator(self, estimator_settings=None):
        if estimator_settings:
            self.estimator = get_estimator(estimator_settings)

    def predict(self, gmt=None):
        if gmt:
            self.annotations['prediction_type'] = 'hypothesis'
            self.annotations['gmt'] = gmt.suffix
        else:
            self.annotations['prediction_type'] = 'background'
            self.annotations['uuid_time_low'] = self.uuid.time_low

    def save_results(self):
        """Saves prediction results (csv) and annotations (json)

        Random gene set predictions are placed in `background_predictions/`
        Gene set predictions are place in `hypothesis_predictions/`
        Each of these gets its own subfolder of `.annotations/`
        """
        outdir_path = get_outdir_path(self.settings)
        if self.annotations['prediction_type'] == 'hypothesis':
            outdir_path += 'hypothesis_predictions/'
        else:
            outdir_path += 'background_predictions/'
        check_or_create_dir(outdir_path)
        outfile_name = self.annotations.get('gmt', str(self.uuid))
        self.results_path = outdir_path + outfile_name + '.csv'
        self.df_result.to_csv(self.results_path, index=False)
        annotations_dir = outdir_path + '.annotations/'
        check_or_create_dir(annotations_dir)
        self.annotations_path = annotations_dir + outfile_name + '.json'
        save_json(self.annotations, self.annotations_path)


class LPOCV(Predictor):
    """Leave-Pair-Out Cross-Validation scheme

    Implements leave-pair-out cross-validation in which each pairs of samples
    are chosen from the dataset followed by training on all but one pair and
    predicting the target variable for the withheld pair.
    """

    def __init__(self, settings=None, dataset=None):
        super().__init__(settings=settings, dataset=dataset)
        self.selected_pairs = []
        self._bcg_predictions = recursivedict()
        self._pairs_list = []

    def predict(self, gmt=None):
        """Performs a background or hypothesis prediction

        Reads in lists of features from a `gmt` or a random feature list for
        background and uses them to fit a model and make a predictions
        """
        super().predict(gmt=gmt)
        if gmt:
            data_cols = self.dataset.data_cols
            for link, _, gene_list, _ in tqdm(gmt.generate(data_cols),
                                              total=len(gmt.gmt)):
                self._build_bcg_predictions(gene_list, link)
        else:
            sampling_range = get_sampling_range(self.settings)
            for k in tqdm(sampling_range):
                gene_list = self.dataset.sample_data_cols(k,
                                                          self.uuid.time_low)
                self._build_bcg_predictions(gene_list, k)
        self._build_df_result()

    def _build_bcg_predictions(self, selected_cols, k):
        """Performs LPOCV and updates the Processor's `_bcg_predictions` dict

        This function is given `selected_cols` corresponding to the selected
        features and `k` a value corresponding to what these selected features
        should be called. In the random case, `k` is a number of selected
        features. In the case where a feature set is supplied via `gmt`, `k`
        corresponds to `gmt.suffix`
        """
        self._get_pairs()
        for pair_index, pair in enumerate(tqdm(self.selected_pairs)):
                pair_ids = (pair[0][0], pair[1][0])
                dataset = self.dataset
                train_test_list = dataset.get_X_y(pair_ids,
                                                  selected_cols=selected_cols)
                X_train, y_train, X_test, _ = train_test_list
                y_train = [int(x) for x in y_train]
                y_train = np.array(y_train)
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

    def _get_pairs(self):
        pairs_proc = PairsProcessor(self.dataset,
                                    self.settings['pairs'])
        self.selected_pairs = pairs_proc.selected_pairs


def get_sampling_range(settings):
    """Returns a list of feature set lengths for background sampling.
    """
    interval_params_list = settings['background']['intervals']
    sampling_range = []
    for params in interval_params_list:
        start = params['start']
        end = params['end']
        step = params['step']
        range_section = list(range(start, end + step, step))
        sampling_range += range_section
    return sampling_range
