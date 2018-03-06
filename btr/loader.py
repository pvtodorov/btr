import json
from .processing_schemes import LPOCV
import synapseclient
import synapseutils
from synapseclient import File, Folder
from .utilities import get_settings_annotations, get_settings_md5
from pprint import pprint


class Loader(object):
    def __init__(self, settings_path=None):
        self.s = None
        self.proc = None
        self._syn = None
        self._settings_path = settings_path
        if self._settings_path:
            with open(self._settings_path) as f:
                self.s = json.load(f)
                self.save_settings_to_synapse(self._settings_path)
        pprint(self.s)

    def get_processor_from_settings(self):
        settings = self.s
        name = settings["processing_scheme"]["name"]
        if name == 'LPOCV':
            self.proc = LPOCV(settings=settings)
        else:
            raise NotImplementedError

    def save_settings_to_synapse(self, settings_path):
        """Saves prediction to Synapse"""
        self._syn = synapseclient.login()
        dirpath = 'run_settings/'
        filename = settings_path.split('/')[-1]
        parent = get_or_create_syn_folder(self._syn,
                                          dirpath,
                                          self.s['project_synid'])
        file = File(path=dirpath + filename, parent=parent)
        annotations = {'btr_file_type': 'settings',
                       'settings_md5': get_settings_md5(self.s)}
        file.annotations = annotations
        file = self._syn.store(file, forceVersion=False)

    def save_prediction_to_synapse(self):
        """Saves prediction to Synapse"""
        self._syn = synapseclient.login()
        dirpath = self.proc._outdir_path
        filename = self.proc._outfile_name
        parent = get_or_create_syn_folder(self._syn,
                                          dirpath,
                                          self.s['project_synid'])
        file = File(path=dirpath + filename, parent=parent)
        annotations = get_settings_annotations(self.s)
        annotations['btr_file_type'] = 'prediction'
        if 'background_predictions' in dirpath:
            annotations['prediction_type'] = 'background'
        elif "hypothesis_predictions" in dirpath:
            annotations['prediction_type'] = "hypothesis"
        else:
            raise NotImplementedError
        gmt = self.proc.gmt
        if gmt:
            annotations['gmt'] = gmt.suffix
        file.annotations = annotations
        file = self._syn.store(file)


def get_or_create_syn_folder(syn, dirpath, project_synid):
    dirs = [x for x in dirpath.split('/') if len(x) > 0]
    parent_obj = syn.get(project_synid)
    while len(dirs) > 0:
        d = dirs.pop(0)
        folder = Folder(d, parent=parent_obj)
        folder = syn.store(folder)
        parent_obj = folder
    return parent_obj
