import argparse
from .loader import Loader


def predict():
    task = 'predict'
    parser = argparse.ArgumentParser()
    parser.add_argument("settings_path", help="settings as JSON")
    parser.add_argument("-i", "--iterations", help="iterations for the script",
                        required=False, default="1")
    parser.add_argument("-g", "--gmt_path",
                        help="path to file or folder of txts", required=False)
    parser.add_argument("-b", "--background_params_path",
                        help="path to background", required=False)
    args = parser.parse_args()
    settings_path = args.settings_path
    iterations = int(args.iterations)
    gmt_path = args.gmt_path
    background_params_path = args.background_params_path
    for i in range(0, iterations):
        loader = Loader(settings_path=settings_path)
        loader.load_processor(task)
        if background_params_path:
            loader.load_background_params(background_params_path)
            loader.proc.predict(background_params=loader.background_params)
        elif gmt_path:
            loader.load_gmt(gmt_path)
            loader.proc.predict(gmt=loader.gmt)
        loader.save(task)


def score():
    task = 'score'
    parser = argparse.ArgumentParser()
    parser.add_argument("settings_path", help="settings as JSON")
    parser.add_argument("-g", "--gmt_path",
                        help="path to file or folder of txts", required=False)
    args = parser.parse_args()
    settings_path = args.settings_path
    gmt_path = args.gmt_path
    loader = Loader(settings_path=settings_path)
    loader.load_dataset(task)
    loader.load_processor(task)
    if gmt_path:
        loader.load_gmt(args.gmt_path)
    loader.proc.get_y_dict(loader.dataset)
    loader.proc.get_score(gmt=loader.gmt)
    loader.save(task)


def stats():
    task = 'stats'
    parser = argparse.ArgumentParser()
    parser.add_argument("settings_path", help="settings as JSON")
    parser.add_argument("-g", "--gmt_path",
                        help="path to file or folder of txts", required=False)
    args = parser.parse_args()
    settings_path = args.settings_path
    gmt_path = args.gmt_path
    loader = Loader(settings_path=settings_path)
    loader.load_dataset(task)
    loader.load_processor(task)
    loader.load_gmt(args.gmt_path)
    print(loader.gmt.suffix)
    loader.proc.get_stats(gmt=loader.gmt, dataset=loader.dataset)
    loader.save(task)
