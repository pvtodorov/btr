import synapseclient
import synapseutils
from synapseclient import File, Folder
from .utilities import load_json, calc_content_md5
from pathlib import Path
import os
from tqdm import tqdm


class SynClient(object):
    def __init__(self):
        self.syn = None
        self.parent_synid = None
        pass

    def login(self):
        self.syn = synapseclient.login()

    def download_file(self, synid):
        pass

    def sync_folder(self, dirpath, direction, overwrite=False):
        if direction == 'up':
            self._upload_folder(dirpath, overwrite=overwrite)
        if direction == 'down':
            raise NotImplementedError

    def _upload_folder(self, dirpath, overwrite=False):
        file_paths = _get_file_paths(dirpath)
        file_paths = [x for x in file_paths
                      if x[0].suffix in ['.csv', '.json']]
        for file_path in tqdm(file_paths):
            self.upload_file(file_path, overwrite=overwrite)

    def upload_file(self, file_path, overwrite=False):
        """Saves file to Synapse"""
        fp = file_path[0]
        ap = file_path[1]
        local_md5 = calc_content_md5(fp)
        parent = self.get_or_create_folder(fp.parent,
                                           self.parent_synid)
        file = File(path=str(fp), parent=parent)
        annotations = load_json(ap)
        annotations = {k.replace('[', 'LFTB').replace(']', 'RGTB'): v
                       for k, v in annotations.items()}
        file.annotations = annotations
        file = self.get_or_create_entity(file,
                                         local_md5=local_md5,
                                         overwrite=overwrite)

    def get_or_create_folder(self, dirpath, parent_synid):
        dirs = dirpath.parts
        folder_synid = parent_synid
        for d in dirs:
            folder = Folder(d, parent=folder_synid)
            folder_synid = self.get_or_create_entity(folder)
        return folder_synid

    def get_or_create_entity(self, entity, local_md5=None,
                             overwrite=False, returnid=True):
        try:
            print('Attempting to get entity "' + entity.name + '".')
            entity_r = self.syn.get(entity, downloadFile=False)
            print('Entity "' + entity_r.name + '" found.')
            if local_md5:
                file_handle = entity_r.__dict__.get('_file_handle', {})
                e_md5 = file_handle.get('contentMd5')
                if local_md5 != e_md5:
                    raise EntityHashMismatch
        except TypeError:
            print('Not found. Attempting to create entity.')
            entity_r = self.syn.store(entity)
        except EntityHashMismatch:
            if overwrite:
                entity_r = self.syn.store(entity)
                print("Overwriting.")
            else:
                print("Overwrite set to `False`.")
        if returnid:
            return entity_r.id
        else:
            return entity_r

    def get_synapse_dict(self, parent_synid):
        walked = synapseutils.walk(self.syn, parent_synid)
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


class EntityHashMismatch(Exception):
    """Local entity and remote entity have different hashes"""
    pass


def _get_file_paths(directory):
    file_paths = []
    walked = os.walk(directory)
    contents = [x for x in walked]
    for dirpath, dirnames, filenames in contents:
        dirpath_p = Path(dirpath)
        if dirpath_p.name != '.annotations':
            for filename in filenames:
                file_p = Path(dirpath, filename)
                anno_p = Path(file_p.parent, '.annotations',
                              file_p.stem + '.json')
                file_paths.append((file_p, anno_p))
    return file_paths
