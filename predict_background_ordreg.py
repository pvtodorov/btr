import argparse
import json
import os
from msbb_functions import *
from mord.threshold_based import LogisticAT
import uuid
import datetime


if __name__ == '__main__':
    dt_start_0 = datetime.datetime.utcnow()
    print(dt_start_0, 'program started')
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
    subset = settings['subset']
    pair_col = settings['pair_col']
    interval = settings['interval']
    max_cols = settings['max_cols']

    data = load_msbb_data(infile)
    data = data[data[subset_col] == subset]
    data_cols = get_data_cols(data, settings['meta_cols'])

    pairs_list = build_pairs_list(data, subset_col, subset, target, transform=digitize_Braak_scores)
    assert(len(sorted(list(set([sample[1] for pair in pairs_list for sample in pair])))) == 3)
    selected_pairs = get_pairs(pairs_list,
                               shuffle=True,
                               seed=47,
                               sample_once=False)

    sampling_range = list(range(10, 510, 10))  # + list(range(500, 1000, 50))

    for i in range(0, iterations):
        df_out = pd.DataFrame()
        predictions_list = []
        bcg_predictions = recursivedict()
        for k in tqdm(sampling_range):
            selected_cols = sample_data_cols(data_cols, k)
            dt_range_start = datetime.datetime.utcnow()
            print(dt_range_start, 'starting range', k)
            for pair_index, pair in enumerate(tqdm(selected_pairs)):
                pair_ids = (pair[0][0], pair[1][0])
                df_train, df_test = get_train_test_df(data, pair_col, pair_ids)
                X_train, y_train = get_X_y(df_train, target, selected_cols)
                y_train = [int(x) for x in y_train]
                y_train = np.array(y_train)
                y_train = digitize_Braak_scores(y_train)
                X_test, y_test = get_X_y(df_test, target, selected_cols)
                e = LogisticAT()
                e = e.fit(X_train, y_train)
                predictions = e.predict(X_test)
                sub_ids = df_test['ID'].tolist()
                for i_s, p in zip(sub_ids, predictions):
                    bcg_predictions[pair_index][i_s][k] = p
            time_diff = datetime.datetime.utcnow() - dt_range_start
            time_diff = time_diff.seconds
            print('range took', time_diff, 's   for ', k, 'columns')
            print(datetime.datetime.utcnow(), 'end range', k)
        for pair_index, p in bcg_predictions.items():
            df_out_t = pd.DataFrame(p)
            df_out_t = df_out_t.transpose()
            df_out_t = df_out_t.reset_index()
            df_out_t = df_out_t.rename(columns={'index': 'ID'})
            df_out_t['pair_index'] = pair_index
            df_out_t_cols = df_out_t.columns.tolist()
            df_out_t_cols = sorted([x for x in df_out_t_cols
                                    if x not in ['ID', 'pair_index']])
            df_out_t_cols = ['ID', 'pair_index'] + df_out_t_cols
            df_out_t = df_out_t[df_out_t_cols]
            df_out = df_out.append(df_out_t)

        outfile = str(uuid.uuid4())
        outfolder = outfolder + '/' + subset + '/'
        check_or_create_dir(outfolder)
        df_out.to_csv(outfolder + '/' + outfile + '.csv', index=False)

        print('entire run takes:',
              (datetime.datetime.utcnow() - dt_start_0).seconds,
              's')
