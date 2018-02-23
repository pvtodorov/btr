import uuid
import os
from collections import defaultdict
import numpy as np


def recursivedict():
    """ Creates a dict of dicts as deep as needed.
    """
    return defaultdict(recursivedict)


def check_or_create_dir(path):
    """ Check if a folder exists. If it doesn't create it.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_outdir_path(settings, gmt=None, sep='-'):
    dset_name = settings["dataset"]["name"]
    filter_name = ""
    if settings['dataset'].get("filter"):
        filter_name = sep + settings["dataset"]["filter"]["name"]
    scheme_name = settings["processing_scheme"]["name"]
    subset_name = settings["processing_scheme"]["subset"]
    subset_name = subset_name.replace('/', '-')
    est_name = settings["estimator"]["name"]
    if settings['misc'].get('tag'):
        misc_tag = sep + settings["misc"]["tag"]
    outdir_path = (dset_name + filter_name + sep +
                   scheme_name + sep + subset_name + sep +
                   est_name +
                   misc_tag + '/')
    return outdir_path


def get_outfile_name(gmt=None):
    if gmt:
        outfile_name = gmt.suffix
    else:
        outfile_name = str(uuid.uuid4())
    outfile_name = outfile_name + '.csv'
    print('outfile name: ' + outfile_name)
    return outfile_name


def digitize_labels(y, transform_thresholds):
    """ Take Braak scores and digitize them such that:
    [0, 1, 2, 3] -> 0
    [4, 5, 6]    -> 1
    """
    y_bin = np.digitize(y, transform_thresholds) - 1
    return y_bin
