import uuid
import os
from collections import defaultdict
import json
import subprocess
import importlib
import hashlib


def recursivedict():
    """ Creates a dict of dicts as deep as needed.
    """
    return defaultdict(recursivedict)


def check_or_create_dir(path):
    """ Check if a folder exists. If it doesn't create it.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_settings_md5(settings):
    """Produces an md5 hash of the settings dict"""
    settings_str = json.dumps(settings, allow_nan=False, sort_keys=True)
    h = hashlib.md5()
    h.update(settings_str.encode())
    return h.hexdigest()


def get_outdir_path(settings):
    """ Assembles the path of the output directory from the settings file
    """
    outdir_path = ('Runs/' + get_settings_md5(settings) + '/')
    return outdir_path


def get_uuid():
    return uuid.uuid4()


def get_outfile_name(name_base):
    """ Returns a name for a file. If a GMT object is given as input, uses
    `gmt.suffix` to produce a name for the file. If not, the file is understood
    to be a background file and a uuid is returned.
    """
    outfile_name = name_base + '.csv'
    print('outfile name: ' + outfile_name)
    return outfile_name


def get_settings_annotations(settings):
    annotations = {}
    # package details
    package_path = importlib.util.find_spec('btr').origin[:-15]
    origin = subprocess.check_output(["git", "-C", package_path, "config",
                                      "--get", "remote.origin.url"]).decode()
    annotations['github_origin'] = origin
    commit_hash = subprocess.check_output(["git", "-C", package_path,
                                           "rev-parse", "HEAD"])
    commit_hash = commit_hash.strip().decode()
    annotations['github_commit_hash'] = commit_hash
    # settings details
    annotations['settings_md5'] = get_settings_md5(settings)
    # dataset details
    ds = settings['dataset']
    annotations['dataset_name'] = ds['name']
    annotations['dataset_filepath'] = ds['filepath']
    annotations['dataset_meta_columns'] = \
        "".join(json.dumps(ds['meta_columns'],
                           allow_nan=False,
                           sort_keys=True))
    annotations['dataset_target'] = ds['target']
    annotations['dataset_ID_column'] = ds['ID_column']
    annotations['dataset_confounder'] = ds.get(['confounder'])
    annotations['dataset_transforms'] = json.dumps(False)
    if ds.get('transforms'):
        for i, f in enumerate(ds['transforms']):
            for k, v in f:
                annotations['dataset_transform_' + str(i) + '_' + str(k)] = \
                    json.dumps(f[v], allow_nan=False, sort_keys=True)
    # processor details
    ps = settings['processor']
    annotations['processor_name'] = ps['name']
    # estimator
    es = ps['estimator']
    annotations['processor_estimator_name'] = ps['estimator']['name']
    annotations['processor_estimator_params'] = \
        json.dumps(es['estimator_params'],
                   allow_nan=False, sort_keys=True)
    annotations['processor_estimator_call'] = \
        es['settings'].get("call", "class")
    # background
    bs = ps["background"]
    for i, p in enumerate(bs['intervals']):
        annotations['background_params_interval_' + str(i)] = \
            json.dumps(p, allow_nan=False, sort_keys=True)
    # lpocv
    if annotations['processor_name'] == 'LPOCV':
        prs = ps['pairs']
        annotations['processor_pairs_shuffle_samples'] = prs['shuffle_samples']
        annotations['processor_pairs_seed'] = prs['shuffle_seed']
        for i, f in enumerate(prs['steps']):
            annotations['processor_pairs_steps_' + str(i)] = f['operation']
            for k, v in f:
                annotations['processor_pairs_steps_' +
                            str(i) + '_' + str(k)] = \
                    json.dumps(f[v], allow_nan=False, sort_keys=True)
    # misc
    annotations['misc'] = "".join(json.dumps(settings['misc']))
    return annotations
