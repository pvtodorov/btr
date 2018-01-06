import argparse
import json
import os
from msbb_functions import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("settings_path", help="settings and parameters json")
    parser.add_argument("iterations", help="how many times to run predict")
    args = parser.parse_args()
    settings_path = args.settings_path
    iterations = int(args.iterations)
    with open(settings_path) as f:
        settings = json.load(f)

    infile = settings['infile']
    outfolder = settings['outfolder']

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    target = settings['target']
    subset_col = settings['subset_col']
    subsets = settings['subsets']

    df = load_msbb_data(infile)
    data_cols = get_data_cols(df, settings['meta_cols'])

    for i in range(0, iterations):
        bcg_predictions = gen_background_predictions(df, target, data_cols,
                                                     subset_col, subsets,
                                                     interval=10,
                                                     max_cols=1000)
        save_predictions(bcg_predictions, outfolder)
