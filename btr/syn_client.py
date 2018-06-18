import synapseclient
import synapseutils
from synapseclient import File, Folder
from .utilities import load_json, get_settings_md5


class SynClient(object):
    def __init__(self):
        self.syn = None
        self.project_synid = None
        pass

    def login(self):
        self.syn = synapseclient.login()

    def download_file(self, synid):
        pass

    def sync_settings(self, settings_path, overwrite=False):
        """Saves settings to Synapse"""
        settings = load_json(settings_path)
        self.project_synid = settings['project_synid']
        parent = self.get_or_create_folder(self.syn,
                                           'run_settings/',
                                           self.project_synid)
        localfile = File(path=settings_path, parent=parent)
        remotefile = self.get_or_create_entity(localfile,
                                               skipget=False,
                                               returnid=False)
        md5 = get_settings_md5(settings)
        if [md5] == remotefile.annotations.get('settings_md5'):
            print('Local settings file and remote have the same md5 hashes.')
        else:
            print('Local settings file and remote have DIFFERENT md5 hashes.')
            if overwrite:
                print('Overwriting remote.')
                file = self.syn.store(localfile)
                annotations = {'type': 'settings',
                               'settings_md5': md5}
                file.annotations = annotations
                file = self.get_or_create_entity(file,
                                                 skipget=overwrite)
            else:
                print('Abort sync.')
                raise synapseclient.exceptions.SynapseError

    def upload_file(self, file_path):
        """Saves prediction to Synapse"""
        file_str, file_dir, annotations_path = get_paths(file_path)
        parent = self.get_or_create_folder(self.syn,
                                           file_dir,
                                           self.project_synid)
        file = File(path=file_path, parent=parent)
        file.annotations = load_json(annotations_path)
        file = self.get_or_create_entity(file,
                                         skipget=True,
                                         returnid=False)

    def get_or_create_folder(self, dirpath, project_synid, max_attempts=10,
                             create=True):
        dirs = [x for x in dirpath.split('/') if len(x) > 0]
        folder_synid = project_synid
        for d in dirs:
            folder = Folder(d, parent=folder_synid)
            folder_synid = self.get_or_create_entity(folder)
        return folder_synid

    def get_or_create_entity(self, entity, max_attempts=10,
                             create=True, skipget=False, returnid=True):
        attempts = 1
        while attempts <= max_attempts:
            try:
                if skipget:
                    print('Writing without checking for entity.')
                    raise TypeError
                else:
                    print('Attempting to get entity "' + entity.name + '".')
                    entity = self.syn.get(entity)
                    entity_synid = entity.id
                    print('Entity "' + entity.name + '" found.')
                    break
            except TypeError:
                try:
                    print('Attempting to create entity.')
                    if create:
                        entity = self.syn.store(entity)
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

    def get_synapse_dict(self, project_synid):
        walked = synapseutils.walk(self.syn, project_synid)
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


def get_paths(file_path):
    file_str = file_path.split('/')[-1][:-4]
    file_dir = file_path[:-(len(file_str) + 4)]
    annotations_path = file_dir + '.annotations/' + \
        file_str + '.json'
    return file_str, file_dir, annotations_path
