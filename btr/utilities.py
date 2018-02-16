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


def get_outdir_path(settings, gmt=None):
    dset_name = settings["dataset"]["name"]
    filter_name = ""
    if settings['dataset'].get("filter"):
        filter_name = "/" + settings["dataset"]["filter"]["name"]
    scheme_name = settings["processing_scheme"]["name"]
    subset_name = settings["processing_scheme"]["subset"]
    est_name = settings["estimator"]["name"]
    prediction_type = "background_predictions"
    if gmt:
        prediction_type = "geneset_predictions"
    outdir_path = (dset_name + filter_name + '/' +
                   scheme_name + '/' + subset_name + '/' +
                   est_name + '/' +
                   prediction_type + '/')
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
