import uuid
import os
from collections import defaultdict
import numpy as np
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


def get_outfile_name(gmt=None):
    """ Returns a name for a file. If a GMT object is given as input, uses
    `gmt.suffix` to produce a name for the file. If not, the file is understood
    to be a background file and a uuid is returned.
    """
    if gmt:
        outfile_name = gmt.suffix
    else:
        outfile_name = str(uuid.uuid4())
    outfile_name = outfile_name + '.csv'
    print('outfile name: ' + outfile_name)
    return outfile_name


def digitize_labels(y, transform_thresholds):
    """ Take Braak scores and digitize them according to thresholds
    """
    y_bin = np.digitize(y, transform_thresholds) - 1
    return y_bin


def get_settings_annotations(settings):
    annotations = {}
    package_path = importlib.util.find_spec('btr').origin[:-15]
    origin = subprocess.check_output(["git", "-C", package_path, "config",
                                      "--get", "remote.origin.url"]).decode()
    annotations['github_origin'] = origin
    commit_hash = subprocess.check_output(["git", "-C", package_path,
                                           "rev-parse", "HEAD"])
    commit_hash = commit_hash.strip().decode()
    annotations['github_commit_hash'] = commit_hash
    annotations['settings_md5'] = get_settings_md5(settings)
    annotations['dataset_name'] = settings['dataset']['name']
    annotations['dataset_filepath'] = settings['dataset']['filepath']
    annotations['dataset_meta_columns'] = \
        "".join(json.dumps(settings['dataset']['meta_columns'],
                           allow_nan=False,
                           sort_keys=True))
    annotations['dataset_target'] = settings['dataset']['target']
    annotations['dataset_ID_column'] = settings['dataset']['ID_column']
    annotations['dataset_filter_name'] = json.dumps(False)
    if settings['dataset'].get('filter'):
        annotations['dataset_filter_name'] = \
            settings['dataset']['filter']['name']
        annotations['dataset_filters_num'] = \
            len(settings['dataset']['filter']['filters'])
        for i, f in enumerate(settings['dataset']['filter']['filters']):
            annotations['dataset_filter_column_' + str(i)] = f['column']
            annotations['dataset_filter_values_' + str(i)] = \
                json.dumps(f['values'], allow_nan=False, sort_keys=True)
    annotations['estimator_name'] = \
        settings['estimator']['name']
    annotations['estimator_params'] = \
        json.dumps(settings['estimator']['estimator_params'],
                   allow_nan=False, sort_keys=True)
    annotations["estimator_call"] = \
        settings["estimator"].get("call", "class")
    annotations['processing_scheme_name'] = \
        settings['processing_scheme']['name']
    if annotations['processing_scheme_name'] == 'LPOCV':
        annotations['processing_scheme_subset_col'] = \
            settings['processing_scheme']['subset_col']
        annotations['processing_scheme_subset'] = \
            settings['processing_scheme']['subset']
        annotations['processing_scheme_pair_col'] = \
            settings['processing_scheme']['pair_col']
        annotations['processing_scheme_transform_labels'] = \
            json.dumps(settings['processing_scheme']['transform_labels'])
        annotations['processing_scheme_pair_settings_shuffle'] = \
            json.dumps(settings['processing_scheme']
                               ['pair_settings']
                               ['shuffle'])
        annotations['processing_scheme_pair_settings_seed'] = \
            settings['processing_scheme']['pair_settings']['seed']
        annotations['processing_scheme_pair_settings_sample_once'] = \
            json.dumps(settings['processing_scheme']
                               ['pair_settings']
                               ['sample_once'])
    annotations['background_params_intervals'] = \
        len(settings['background_params']['intervals'])
    for i, p in enumerate(settings['background_params']['intervals']):
        annotations['background_params_interval_' + str(i)] = \
            json.dumps(p, allow_nan=False, sort_keys=True)
    annotations['misc'] = "".join(json.dumps(settings['misc']))
    return annotations
