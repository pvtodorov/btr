import argparse
import json
import os
from msbb_functions import *


def gen_background_predictions(df, target, data_cols,
                               subset_col, subsets,
                               interval=10, max_cols=1000):
    """ Generate binary classification predictions accross subsets.
    """
    bcg_predictions = recursivedict()
    for subset in subsets:
        df_train, df_test = get_train_test_df(df, subset_col, [subset])
        for k in tqdm(range(10, max_cols + interval, interval)):
            selected_cols = sample_data_cols(data_cols, k)
            X_train, y_train = get_X_y(df_train, target, selected_cols)
            y_train = binarize_Braak_scores(y_train)
            X_test, y_test = get_X_y(df_test, target, selected_cols)
            rf = fit_RF(X_train, y_train, 'classification')
            predictions = predict_RF(rf, X_test)
            sub_ids = df_test['ID'].tolist()
            for id, p in zip(sub_ids, predictions):
                bcg_predictions[subset[0]][id][k] = p
    return bcg_predictions


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
