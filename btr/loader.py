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
        md5 = get_settings_md5(self.s)
        query_str = 'SELECT * FROM file WHERE settings_md5==\"' + \
                    md5 + '\" AND btr_file_type==\"settings\"'
        q = self._syn.chunkedQuery(query_str)
        qlist = [x for x in q]
        if len(qlist) == 1:
            return
        else:
            dirpath = 'run_settings/'
            filename = settings_path.split('/')[-1]
            parent = get_or_create_syn_folder(self._syn,
                                              dirpath,
                                              self.s['project_synid'])
            file = File(path=dirpath + filename, parent=parent)
            annotations = {'btr_file_type': 'settings',
                           'settings_md5': get_settings_md5(md5)}
            file.annotations = annotations
            file = self._syn.store(file)

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
    parent_synid = project_synid
    for d in dirs:
        attempts = 1
        while attempts <= max_attempts:
            try:
                folder = Folder(d, parent=parent_synid)
                folder = syn.get(folder)
                parent_synid = folder.id
                break
            except TypeError:
                try:
                    print('folder ' + d + ' not found. attempting to create')
                    if create:
                        folder = syn.store(folder)
                        parent_synid = folder.id
                        break
                    else:
                        print('create set to False. folder not created.')
                        break
                except synapseclient.exceptions.SynapseHTTPError:
                    print('SynapseHTTPError. Retrying. Attempt ' + attempts)
                    attempts += 1
                    continue
    return parent_synid
