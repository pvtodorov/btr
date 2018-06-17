import json
from .processing_schemes import LPOCV
import synapseclient
import synapseutils
from synapseclient import File, Folder
from .utilities import (flatten_settings, get_settings_annotations)
from .gmt import GMT
from .dataset import Dataset
from .scorer import ScoreLPOCV
from pprint import pprint


class Loader(object):
    def __init__(self,
                 settings_path=None, gmt_path=None,
                 use_synapse=True, syn_settings_overwrite=False,
                 pprint_settings=False):
        self.settings = None
        self.dataset = None
        self.proc = None
        self.gmt = None
        self.syn = None
        self.settings_path = settings_path
        self.gmt_path = gmt_path
        # if a settings_path is provided, automatically load settings
        if self.settings_path:
            self.load_settings(self.settings_path)
        # if use_synapse True, automatically login
        if use_synapse:
            self.login_synapse(overwrite=syn_settings_overwrite)
        if pprint_settings:
            pprint(self.settings)

    def load_settings(self, settings_path):
        with open(self.settings_path) as f:
            self.settings = json.load(f)

    def load_gmt(self, gmt_path):
        if gmt_path:
            self.gmt = GMT(gmt_path)

    def login_synapse(self, overwrite=False):
        self.syn = synapseclient.login()
        self.save_settings_to_synapse(self.settings_path,
                                      overwrite=overwrite)

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
        self.settings['processor'] = self.settings["processor"]
        name = self.settings['processor']["name"]
        if name == 'LPOCV':
            if task == "predict":
                self.proc = LPOCV(settings=self.settings['processor'],
                                  dataset=self.dataset)
            elif task == 'score':
                self.proc = ScoreLPOCV(settings=self.settings)
            elif task == "stats":
                self.proc = ScoreLPOCV(settings=self.settings)
        else:
            raise NotImplementedError

    def get_annotations(self):
        self.annotations = {}
        self.annotations = get_settings_annotations(self.settings)
        self.annotations.update(get_btr_version_info())
        to_update = {'dataset.': self.dataset,
                     'processor.': self.proc}
        for prefix, obj in to_update.items():
            flat_settings = flatten_settings(obj.settings, prefix)
            self.annotations.update(flat_settings)
            flat_annotations = flatten_settings(obj.annotations, prefix)
            self.annotations.update(flat_annotations)

    def save_settings_to_synapse(self, settings_path, overwrite=False):
        """Saves prediction to Synapse"""
        if not self.syn:
            print('Not logged into synapse.')
            return
        parent = get_or_create_syn_folder(self.syn,
                                          'run_settings/',
                                          self.settings['project_synid'])
        localfile = File(path=settings_path, parent=parent)
        remotefile = get_or_create_syn_entity(localfile, self.syn,
                                              skipget=False,
                                              returnid=False)
        self.get_annotations()
        md5 = self.annotations['settings_md5']
        if [md5] == remotefile.annotations.get('settings_md5'):
            print('Local settings file and remote have the same md5 hashes.')
        else:
            print('Local settings file and remote have DIFFERENT md5 hashes.')
            if overwrite:
                print('Overwriting remote.')
                file = self.syn.store(localfile)
                annotations = {'btr_file_type': 'settings',
                               'settings_md5': md5}
                file.annotations = annotations
                file = get_or_create_syn_entity(file, self.syn,
                                                skipget=overwrite)
            else:
                print('Overwrite disabled. Remote unchanged. Local unchanged.')
                raise synapseclient.exceptions.SynapseError

    def save_prediction_to_synapse(self):
        """Saves prediction to Synapse"""
        if not self.syn:
            print('Not logged into synapse')
            return
        dirpath = self.proc._outdir_path
        filename = self.proc._outfile_name
        parent = get_or_create_syn_folder(self.syn,
                                          dirpath,
                                          self.settings['project_synid'])
        file = File(path=dirpath + filename, parent=parent)
        annotations = get_settings_annotations(self.settings)
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
        file = get_or_create_syn_entity(file, self.syn,
                                        skipget=True,
                                        returnid=False)


def get_synapse_dict(syn, project_synid):
    walked = synapseutils.walk(syn, project_synid)
    contents = [x for x in walked]
    contents_dict = {}
    project_name = contents[0][0][0]
    contents_dict[''] = contents[0][0][1]
    for c in contents:
        base = c[0]
        folders = c[1]
        files = c[2]
        for folder in folders:
            folder_path = "/".join([base[0], folder[0], ''])
            folder_path = folder_path.replace(project_name + '/', '')
            syn_id = folder[1]
            contents_dict[folder_path] = syn_id
        for file in files:
            file_path = "/".join([base[0], file[0]])
            file_path = file_path.replace(project_name + '/', '')
            syn_id = file[1]
            contents_dict[file_path] = syn_id
    return contents_dict


def get_or_create_syn_folder(syn, dirpath, project_synid, max_attempts=10,
                             create=True):
    dirs = [x for x in dirpath.split('/') if len(x) > 0]
    folder_synid = project_synid
    for d in dirs:
        folder = Folder(d, parent=folder_synid)
        folder_synid = get_or_create_syn_entity(folder, syn)
    return folder_synid


def get_or_create_syn_entity(entity, syn, max_attempts=10,
                             create=True, skipget=False, returnid=True):
    attempts = 1
    while attempts <= max_attempts:
        try:
            if skipget:
                print('Writing without checking for entity.')
                raise TypeError
            else:
                print('Attempting to get entity "' + entity.name + '".')
                entity = syn.get(entity)
                entity_synid = entity.id
                print('Entity "' + entity.name + '" found.')
                break
        except TypeError:
            try:
                print('Attempting to create entity.')
                if create:
                    entity = syn.store(entity)
                    entity_synid = entity.id
                    break
                else:
                    print('Create set to False. Entity not created.')
                    break
            except synapseclient.exceptions.SynapseHTTPError:
                print('SynapseHTTPError. Retrying. Attempt ' + attempts)
                attempts += 1
                continue
    if returnid:
        return entity_synid
    else:
        return entity
