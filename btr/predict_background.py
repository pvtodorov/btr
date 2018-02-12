import pandas as pd
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
    df = pd.read_table(infile)
    data_cols = get_data_cols(df, settings['meta_cols'])

    target = settings['target']
    subset = settings['subset']

    # make sure df target variable doesn't have NaNs
    df = df[(df[target].notnull())]
    # select a particular subset of the dataset based on BrodmannArea
    df = df[df['BrodmannArea'] == subset]
    # drop duplicate IDs, taking only the first time the ID occurs
    df = df.drop_duplicates(subset='ID', keep='first')
    data_cols = get_data_cols(df, settings['meta_cols'])
    print(df.shape)

    for i in range(0, iterations):
        oob_scores = gen_background_performance(df,
                                                target,
                                                data_cols)
        save_oob_scores(oob_scores, outfolder)
