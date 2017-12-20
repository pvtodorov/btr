import pandas as pd
from tqdm import tqdm
import argparse
import json
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("settings_path", help="settings and parameters json")
    args = parser.parse_args()
    settings_path = args.settings_path
    with open(settings_path) as f:
        settings = json.load(f)

    outfolder = settings['outfolder']
    bg_runs = os.listdir(outfolder)
    bg_runs = [x for x in bg_runs if '.csv' in x]
    aggregate_runs = pd.DataFrame()
    for fn in tqdm(bg_runs):
        df = pd.read_csv(outfolder + '/' + fn)
        df = df.set_index('n_features').transpose().reset_index(drop=True)
        df.columns.name = None
        aggregate_runs = aggregate_runs.append(df)
    aggregate_runs.to_csv(outfolder + '.csv', index=False)
