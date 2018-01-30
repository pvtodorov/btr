import pandas as pd
import uuid
import os
from collections import defaultdict


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
    dset_name = self.s["dataset"]["name"]
    scheme_name = self.s["processing_scheme"]["name"]
    subset_name = self.s["processing_scheme"]["subset"]
    est_name = self.s["estimator"]["name"]
    prediction_type = "background_predictions"
    if gmt:
        prediction_type = "geneset_predictions"
    outdir_path = (dset_name + '/' +
                    scheme_name + '/' + subset_name + '/' +
                    est_name + '/' +
                    prediction_type + '/')
    return outdir_path

def get_outfile_name(settings, gmt=None):
    if gmt:
        outfile_name = gmt.suffix
    else:
        outfile_name = str(uuid.uuid4())
    outfile_name = outfile_name + '.csv'
    return outfile_name
