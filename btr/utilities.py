import uuid
import os
from collections import defaultdict
import json
import subprocess
import importlib
import hashlib
import morph


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
    flat_settings = flatten_settings(settings)
    settings_str = json.dumps(flat_settings, allow_nan=False, sort_keys=True)
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


def flatten_settings(settings, prefix=''):
    flat_settings = morph.flatten(settings)
    flat_settings = {prefix + k: v for k, v in flat_settings.items()}
    return flat_settings


def get_btr_version_info():
    version_info = {}
    package_path = importlib.util.find_spec('btr').origin[:-15]
    origin = subprocess.check_output(["git", "-C", package_path, "config",
                                      "--get", "remote.origin.url"]).decode()
    version_info['btr.github_origin'] = origin
    commit_hash = subprocess.check_output(["git", "-C", package_path,
                                           "rev-parse", "HEAD"])
    commit_hash = commit_hash.strip().decode()
    version_info['btr.github_commit_hash'] = commit_hash
    return version_info


def get_settings_annotations(settings):
    annotations = {}
    annotations.update(flatten_settings(settings))
    annotations.update({'settings_md5': get_settings_md5(settings)})
    return annotations
