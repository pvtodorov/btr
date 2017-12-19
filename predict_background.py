import pandas as pd
import random
from sklearn.ensemble import RandomForestRegressor
import uuid
from tqdm import tqdm
import argparse
import json
import os


def get_data_cols(df, meta_cols):
    cols = df.columns.tolist()
    data_cols = [x for x in cols if x not in meta_cols]
    return data_cols


def sample_data_cols(data_cols, k):
    sampled_cols = random.sample(data_cols, k)
    return [x for x in data_cols if x in sampled_cols]


def get_X_y(df, target, data_cols):
    X = df.as_matrix(columns=data_cols)
    y = df[target].tolist()
    return X, y


def fit_RF(X, y):
    rf = RandomForestRegressor(n_estimators=100,
                               max_features='auto',
                               n_jobs=4,
                               oob_score=True)
    rf.fit(X, y)
    return rf


def gen_background_performance(df, target, data_cols,
                               interval=10,
                               max_cols=500):
    oob_scores = {}
    for k in tqdm(range(10, max_cols, interval)):
        selected_cols = sample_data_cols(data_cols, k)
        X, y = get_X_y(df, target, selected_cols)
        rf = fit_RF(X, y)
        oob_scores[k] = rf.oob_score_
    return oob_scores


def save_oob_scores(oob_scores, folder):
    d = oob_scores
    scores = pd.DataFrame(list(d.items()), columns=['n_features', 'R2'])
    scores = scores.sort_values('n_features').reset_index(drop=True)
    outfile = str(uuid.uuid4())
    scores.to_csv(folder + '/' + outfile + '.csv', index=False)


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

    df = df[(df[target].notnull())]
    df = df[df['BrodmannArea'] == subset]
    df = df.drop_duplicates(subset='ID', keep='first')
    data_cols = get_data_cols(df, settings['meta_cols'])
    print(df.shape)

    for i in range(0, iterations):
        oob_scores = gen_background_performance(df,
                                                target,
                                                data_cols)
        save_oob_scores(oob_scores, outfolder)
