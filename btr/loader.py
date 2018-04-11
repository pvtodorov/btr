import json
from .processing_schemes import LPOCV
import synapseclient
import synapseutils
from synapseclient import File, Folder
from .utilities import get_settings_annotations, get_settings_md5
from pprint import pprint


class Loader(object):
    def __init__(self, settings_path=None,
                 use_synapse=True, syn_settings_overwrite=False):
        self.s = None
        self.proc = None
        self._settings_path = settings_path
        if self._settings_path:
            with open(self._settings_path) as f:
                self.s = json.load(f)
        self._syn = None
        if use_synapse:
            self._syn = synapseclient.login()
            self.save_settings_to_synapse(self._settings_path,
                                          overwrite=syn_settings_overwrite)
        pprint(self.s)

    def get_processor_from_settings(self):
        settings = self.s
        name = settings["processing_scheme"]["name"]
        if name == 'LPOCV':
            self.proc = LPOCV(settings=settings)
        else:
            raise NotImplementedError

    def save_settings_to_synapse(self, settings_path, overwrite=False):
        """Saves prediction to Synapse"""
        if not self._syn:
            print('Not logged into synapse.')
            return
        parent = get_or_create_syn_folder(self._syn,
                                          'run_settings/',
                                          self.s['project_synid'])
        localfile = File(path=settings_path, parent=parent)
        remotefile = get_or_create_syn_entity(localfile, self._syn,
                                              skipget=False,
                                              returnid=False)
        md5 = get_settings_md5(self.s)
        if [md5] == remotefile.annotations.get('settings_md5'):
            print('Local settings file and remote have the same md5 hashes.')
        else:
            print('Local settings file and remote have DIFFERENT md5 hashes.')
            if overwrite:
                print('Overwriting remote.')
                file = self._syn.store(localfile)
                annotations = {'btr_file_type': 'settings',
                               'settings_md5': md5}
                file.annotations = annotations
                file = get_or_create_syn_entity(file, self._syn,
                                                skipget=overwrite)
            else:
                print('Overwrite disabled. Remote unchanged. Local unchanged.')
                raise synapseclient.exceptions.SynapseError

    def save_prediction_to_synapse(self):
        """Saves prediction to Synapse"""
        if not self._syn:
            print('Not logged into synapse')
            return
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
        file = get_or_create_syn_entity(file, self._syn,
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
