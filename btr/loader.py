from pprint import pprint

from .dataset import Dataset
from .gmt import GMT
from .processing_schemes import LPOCV
from .scorer import ScoreLPOCV
from .utilities import (flatten_settings, get_btr_version_info,
                        get_outdir_path, get_settings_annotations, load_json,
                        save_json, check_or_create_dir)


class Loader(object):
    def __init__(self,
                 settings_path=None,
                 gmt_path=None,
                 pprint_settings=False):
        self.settings = None
        self.dataset = None
        self.proc = None
        self.gmt = None
        self.settings_path = settings_path
        self.gmt_path = gmt_path
        # if a settings_path is provided, automatically load settings
        if self.settings_path:
            self.load_settings(self.settings_path)
        if pprint_settings:
            pprint(self.settings)

    def load_settings(self, settings_path):
        self.settings = load_json(settings_path)

    def load_gmt(self, gmt_path):
        if gmt_path:
            self.gmt = GMT(gmt_path)

    def load_dataset(self, task):
        ds = self.settings['dataset']
        if task == 'predict':
            self.dataset = Dataset(ds)
        elif task == 'score':
            self.dataset = Dataset(ds, usecols=ds['meta_columns'])
        elif task == 'stats':
            self.dataset = Dataset(ds, cols_only=True)
        else:
            raise NotImplementedError

    def load_processor(self, task):
        if task == "predict":
            self.load_predictor()
        elif task == 'score':
            self.load_scorer()
        elif task == "stats":
            raise NotImplementedError
        else:
            raise NotImplementedError

    def load_predictor(self, load_data=True):
        scheme = self.settings['processor']["scheme"]
        if load_data:
            self.load_dataset('predict')
        if scheme == 'LPOCV':
            self.proc = LPOCV(settings=self.settings['processor'],
                              dataset=self.dataset)
        else:
            raise NotImplementedError

    def load_scorer(self):
        scheme = self.settings['processor']["scheme"]
        if scheme == 'LPOCV':
            self.proc = ScoreLPOCV(settings=self.settings)
        else:
            raise NotImplementedError

    def load_statser(self):
        scheme = self.settings['processor']["scheme"]
        if scheme == 'LPOCV':
            pass
        else:
            raise NotImplementedError

    def get_annotations(self):
        self.annotations = {}
        self.annotations = get_settings_annotations(self.settings)
        self.annotations.update(get_btr_version_info())
        to_update = {'dataset.': self.dataset,
                     'processor.': self.proc}
        for prefix, obj in to_update.items():
            flat_annotations = flatten_settings(obj.annotations, prefix)
            self.annotations.update(flat_annotations)

    def save(self, task):
        if task == "predict":
            self._save_results()
        elif task == 'score':
            raise NotImplementedError
        elif task == "stats":
            raise NotImplementedError
        else:
            raise NotImplementedError

    def _save_results(self):
        """Saves prediction results (csv) and annotations (json)

        Random gene set predictions are placed in `background_predictions/`
        Gene set predictions are place in `hypothesis_predictions/`
        Each of these gets its own subfolder of `.annotations/`
        """
        outdir_path = get_outdir_path(self.settings)
        if self.proc.annotations['prediction_type'] == 'hypothesis':
            outdir_path += 'hypothesis_predictions/'
        else:
            outdir_path += 'background_predictions/'
        check_or_create_dir(outdir_path)
        outfile_name = self.proc.annotations.get('gmt', str(self.proc.uuid))
        results_path = outdir_path + outfile_name + '.csv'
        self.proc.df_result.to_csv(results_path, index=False)
        annotations_dir = outdir_path + '.annotations/'
        check_or_create_dir(annotations_dir)
        annotations_path = annotations_dir + outfile_name + '.json'
        self.get_annotations()
        save_json(self.annotations, annotations_path)
