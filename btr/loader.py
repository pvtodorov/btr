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
        self.backgrounds_params = None
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
            self.gmt_path = gmt_path

    def load_background_params(self, background_params_path):
        self.background_params = load_json(background_params_path)

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
            self._save_predictions()
        elif task == 'score':
            self._save_score()
        elif task == "stats":
            raise NotImplementedError
        else:
            raise NotImplementedError

    def _save_predictions(self):
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
        predictions_path = outdir_path + outfile_name + '.csv'
        self.proc.df_result.to_csv(predictions_path, index=False)
        self.get_annotations()
        save_annotations(self.annotations, predictions_path)

    def _save_score(self):
        """Saves prediction results (csv) and annotations (json)

        Random gene set predictions are placed in `background_predictions/`
        Gene set predictions are place in `hypothesis_predictions/`
        Each of these gets its own subfolder of `.annotations/`
        """
        infolder = get_outdir_path(self.settings)
        outfolder = "/".join(infolder.split('/')[:-1] + ['score', ''])
        check_or_create_dir(outfolder)
        file_name = self.proc.annotations.get('gmt', 'background')
        score_path = outfolder + file_name + '_auc.csv'
        self.proc.df.to_csv(score_path, index=False)
        self.get_annotations()
        save_annotations(self.annotations, score_path)


def save_annotations(annotations_dict, filepath):
    output_file = filepath.split('/')[-1]
    output_dir = filepath[:-len(output_file)]
    file_name = output_file[:-4]
    annotations_dir = output_dir + '.annotations/'
    annotations_filepath = annotations_dir + file_name + '.json'
    check_or_create_dir(annotations_dir)
    save_json(annotations_dict, annotations_filepath)
