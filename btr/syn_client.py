import synapseclient
import synapseutils
from synapseclient import File, Folder
from .utilities import load_json, save_json, calc_content_md5, recursivedict
from pathlib import Path
import os
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
import json
from asciitree import LeftAligned
from asciitree.drawing import BoxStyle, BOX_LIGHT


class SynClient(object):
    def __init__(self):
        self.syn = None
        pass

    def login(self):
        self.syn = synapseclient.login(silent=True)

    def sync_folder(self, dirpath, direction, overwrite=False):
        if direction == 'up':
            self._upload_folder(dirpath, overwrite=overwrite)
        if direction == 'down':
            self._download_folder()

    def _upload_folder(self, dirpath, overwrite=False):
        file_paths = _get_file_paths(dirpath)
        for file_path in tqdm(file_paths):
            self.upload_file(file_path, overwrite=overwrite)

    def upload_file(self, file_path, overwrite=False):
        """Saves file to Synapse"""
        fp = file_path['file_p']
        ap = _get_anno_p(fp)
        local_md5 = calc_content_md5(fp)
        parent = self.get_or_create_folder(fp.parent,
                                           self.parent_synid)
        file = File(path=str(fp), parent=parent)
        annotations = load_json(ap)
        annotations = {k.replace('[', 'LFTB').replace(']', 'RGTB'): v
                       for k, v in annotations.items()}
        file.annotations = annotations
        file_r = self.get_or_create_entity(file, returnid=False)
        if overwrite:
            print(local_md5)
            print(file_r.md5)
            if local_md5 != file_r.md5:
                print('mismatch. overwriting.')
                self.syn.store(file)

    def download_file(self, synapse_path, overwrite=False):
        """Pull file from Synapse"""
        synid = synapse_path['synid']
        fp = Path(synapse_path['file_p'])
        ap = _get_anno_p(fp)
        fp.parent.mkdir(parents=True, exist_ok=True)
        ap.parent.mkdir(parents=True, exist_ok=True)
        if fp.exists():
            print('File exists.')
            local_md5 = calc_content_md5(fp)
            entity = self.syn.get(synid, downloadFile=False)
            if local_md5 != entity.md5:
                print('Local and remote md5 do not match.')
                if overwrite:
                    print("Overwriting.")
                    entity = self.syn.get(synid,
                                          downloadFile=True,
                                          downloadLocation=fp.parent,
                                          ifcollision='overwrite.local')
                    anno = entity.annotations
                    anno = {k.replace('LFTB', '[').replace('RGTB', ']'): v[0]
                            for k, v in anno.items()}
                    save_json(anno, ap)
                else:
                    print("Overwrite set to `False`.")
            else:
                print('Local and remote md5 match.')
        else:
            entity = self.syn.get(synid,
                                  downloadFile=True,
                                  downloadLocation=fp.parent,
                                  ifcollision='overwrite.local')
            anno = entity.annotations
            anno = {k.replace('LFTB', '[').replace('RGTB', ']'): v[0]
                    for k, v in anno.items()}
            save_json(anno, ap)

    def get_or_create_folder(self, dirpath, parent_synid):
        dirs = dirpath.parts
        folder_synid = parent_synid
        for d in dirs:
            folder = Folder(d, parent=folder_synid)
            folder_synid = self.get_or_create_entity(folder)
        return folder_synid

    def get_or_create_entity(self, entity, returnid=True):
        try:
            print('Attempting to get entity "' + entity.name + '".')
            entity_r = self.syn.get(entity, downloadFile=False)
            print('Entity "' + entity_r.name + '" found.')
        except TypeError:
            print('Not found. Creating entity.')
            entity_r = self.syn.store(entity)
        if returnid:
            return entity_r.id
        else:
            return entity_r


def init(synid):
    syn_dir = Path('.syn-vc')
    try:
        syn_dir.mkdir()
        syn_config = {'parentId': synid}
        save_json(syn_config, '.syn-vc/config')
        save_json({}, '.syn-vc/file-paths')
    except FileExistsError:
        print('Synapse tracking already initialized.')


def fetch():
    syn_config = load_json('.syn-vc/config')
    project_id = syn_config['parentId']
    local_paths = _get_file_paths('.')
    sc = SynClient()
    sc.login()
    remote_paths = _get_synapse_file_paths(sc.syn, project_id)
    common_paths = build_common_paths(local_paths, remote_paths)
    cp_serializable = {str(k): v for k, v in common_paths.items()}
    save_json(cp_serializable, '.syn-vc/file-paths')


def status(dirpath):
    dirpath = Path(dirpath)
    print(dirpath)
    common_paths = load_json('.syn-vc/file-paths')
    common_paths = {Path(k): v for k, v in common_paths.items()}
    retain_paths = {}
    if dirpath == Path('.'):
        retain_paths = common_paths
    else:
        for cp, meta in common_paths.items():
            if is_subpath(cp, dirpath):
                retain_paths[cp] = meta
    print_diff(retain_paths)


def build_common_paths(local_paths, remote_paths):
    all_paths = [x['file_p'] for x in local_paths]
    all_paths = all_paths + [x['file_p'] for x in remote_paths]
    all_paths = list(set(all_paths))
    common_paths = {}
    for file_p in tqdm(all_paths):
        keys = ['local_md5', 'remote_md5', 'synid']
        cp = {k: None for k in keys}
        lp = [x for x in local_paths if x['file_p'] == file_p]
        if lp:
            assert(len(lp) == 1)
            cp['local_md5'] = lp[0]['md5']
        rp = [x for x in remote_paths if x['file_p'] == file_p]
        if rp:
            assert(len(rp) == 1)
            cp['remote_md5'] = rp[0]['md5']
            cp['synid'] = rp[0]['synid']
        common_paths[file_p] = cp
    return common_paths


def common_paths_to_str(common_paths):
    cp_serializable = {str(k): v for k, v in common_paths.items()}
    comm_path_str = json.dumps(cp_serializable, allow_nan=False, indent=2)
    return comm_path_str


def common_paths_from_str(comm_path_str):
    common_paths = json.loads(comm_path_str)
    common_paths = {Path(k): v for k, v in common_paths.items()}
    return common_paths


def paths_to_tree(file_paths):
    tree = recursivedict()
    for fp in file_paths:
        parents = get_parents(fp)
        t = tree
        while len(parents):
            parent = parents.pop()
            t[parent.name].update({})
            t = t[parent.name]
        # import pdb; pdb.set_trace()
        n_files = len([x for x in file_paths if parent in x.parents])
        t.update({str(n_files) + ' file(s)': {}})
    return tree


def print_diff(common_paths):
    comm_path_str = common_paths_to_str(common_paths)
    common_paths = common_paths_from_str(comm_path_str)
    untracked_files = {k: v for k, v in common_paths.items() if not v['synid']}
    remote_files = {k: v for k, v in common_paths.items()
                    if not v['local_md5']}
    mismatch_files = {k: v for k, v in common_paths.items()
                      if ((None not in [v['local_md5'], v['remote_md5']]) 
                          & (len(set([v['local_md5'], v['remote_md5']]))) > 1)}
    d = {'Untracked files:': untracked_files,
         'Remote files:': remote_files,
         'Mismatched_files:': mismatch_files}
    for k, v in d.items():
        head_str = k + '  ' + str(len(v)) + ' file(s).'
        print(head_str)
        if len(v) > 0:
            tr = LeftAligned(draw=BoxStyle(gfx=BOX_LIGHT, horiz_len=1))
            tree = paths_to_tree(v)
            print('  ' + tr(tree).replace('\n', '\n  '))
        print()
        print()


def _get_file_paths(directory):
    print('Mapping local paths.')
    file_paths = []
    walked = os.walk(directory)
    contents = [x for x in walked]
    for dirpath, dirnames, filenames in contents:
        dirpath_p = Path(dirpath)
        if dirpath_p.name != '.annotations':
            for filename in filenames:
                file_p = Path(dirpath, filename)
                file_path = {'file_p': file_p,
                             'md5': calc_content_md5(file_p)}
                file_paths.append(file_path)
    file_paths = [x for x in file_paths
                  if x['file_p'].suffix in ['.csv', '.json']]
    return file_paths


def _get_synapse_file_paths(syn, parent_synid):
    print('Mapping synapse paths.')
    e = syn.get(parent_synid)
    strip_dir = ''
    if e.concreteType == 'org.sagebionetworks.repo.model.Project':
        strip_dir = e.name
    walked = synapseutils.walk(syn, parent_synid)
    contents = [x for x in walked]
    file_paths = []
    for dirpath, dirnames, filenames in contents:
        for filename in filenames:
            file_p = Path(dirpath[0], filename[0])
            file_p = Path(*[x for x in file_p.parts if x != strip_dir])
            file_path = {'file_p': file_p,
                         'synid': filename[1]}
            file_paths.append(file_path)
    synids = [x['synid'] for x in file_paths]
    se_list = _threaded_syn_get(syn, synids)
    remote_md5_dict = {x.id: x.md5 for x in se_list}
    for file_path in file_paths:
        file_path.update({'md5': remote_md5_dict[file_path['synid']]})
    return file_paths


def _threaded_syn_get(syn, synids, pool_size=12):
    pool = ThreadPool(pool_size)
    se_list = list(tqdm(
        pool.imap(
            lambda synid: syn.get(synid, downloadFile=False), synids),
        total=len(synids)))
    return se_list


def _get_anno_p(file_p):
    anno_p = Path(file_p.parent, '.annotations',
                  file_p.stem + '.json')
    return anno_p


def get_parents(path):
    parents = []
    if path.is_dir():
        parent = path
    else:
        parent = path.parent
    while str(parent) != '.':
        parents.append(parent)
        parent = parent.parent
    return parents


def is_subpath(path_a, path_b):
    parents_a = get_parents(path_a)
    parents_b = get_parents(path_b)
    intersect = [x for x in parents_b if x in parents_a]
    if len(intersect):
        result = True
    else:
        result = False
    return result
