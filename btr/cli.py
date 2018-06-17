import argparse

from .gmt import GMT
from .loader import Loader


def predict():
    task = 'predict'
    parser = argparse.ArgumentParser()
    parser.add_argument("settings_path", help="settings as JSON")
    parser.add_argument("-i", "--iterations", help="iterations for the script",
                        required=False, default="1")
    parser.add_argument("-g", "--gmt_path",
                        help="path to file or folder of txts", required=False)
    parser.add_argument("-o", "--overwrite_settings",
                        help="path to file or folder of txts", required=False,
                        action='store_true')
    args = parser.parse_args()
    settings_path = args.settings_path
    iterations = int(args.iterations)
    overwrite_settings = args.overwrite_settings
    gmt_path = args.gmt_path
    gmt = None
    for i in range(0, iterations):
        if gmt_path:
            gmt = GMT(gmt_path)
        loader = Loader(settings_path=settings_path, use_synapse=True,
                        syn_settings_overwrite=overwrite_settings)
        loader.load_processor(task)
        loader.proc.predict(gmt=gmt)
        loader.proc.save_results()
        loader.save_prediction_to_synapse()


def score():
    task = 'score'
    parser = argparse.ArgumentParser()
    parser.add_argument("settings_path", help="settings as JSON")
    parser.add_argument("-g", "--gmt_path",
                        help="path to file or folder of txts", required=False)
    args = parser.parse_args()
    settings_path = args.settings_path
    loader = Loader(settings_path=settings_path, use_synapse=False,
                    syn_settings_overwrite=True)
    loader.load_dataset(task)
    loader.load_processor(task)
    loader.load_gmt(args.gmt_path)
    loader.proc.get_y_dict(loader.dataset)
    loader.proc.score_LPOCV()


# def stats():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("settings_path", help="settings as JSON")
#     parser.add_argument("-g", "--gmt_path",
#                         help="path to file or folder of txts",
#                         required=False)
#     args = parser.parse_args()
#     settings_path = args.settings_path
#     gmt_path = args.gmt_path
#     scorer = Scorer()
#     scorer.from_settings(settings_path)
#     scorer.score_LPOCV(gmt_path=gmt_path)
#     scorer.get_stats(gmt_path=gmt_path)
