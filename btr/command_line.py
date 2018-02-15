import argparse
from .loader import Loader
from .gmt import GMT
from .score import Scorer


def predict_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("settings_path", help="settings as JSON")
    parser.add_argument("-i", "--iterations", help="iterations for the script",
                        required=False, default="1")
    parser.add_argument("-g", "--gmt_path",
                        help="path to file or folder of txts", required=False)
    args = parser.parse_args()
    settings_path = args.settings_path
    iterations = int(args.iterations)
    gmt_path = args.gmt_path
    gmt = None
    loader = Loader(settings_path)
    proc = loader.processor_from_settings()
    proc.from_settings()
    if gmt_path:
        gmt = GMT(gmt_path)
        proc.predict_gmt(gmt)
        proc.save_results(gmt)
    else:
        for i in range(0, iterations):
            proc.predict_background()
            proc.save_results(gmt)


def score_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("settings_path", help="settings as JSON")
    parser.add_argument("-g", "--gmt_path",
                        help="path to file or folder of txts", required=False)
    args = parser.parse_args()
    settings_path = args.settings_path
    gmt_path = args.gmt_path
    scorer = Scorer()
    scorer.from_settings(settings_path)
    scorer.score_LPOCV(gmt_path=gmt_path)


def stats_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("settings_path", help="settings as JSON")
    parser.add_argument("-g", "--gmt_path",
                        help="path to file or folder of txts", required=False)
    args = parser.parse_args()
    settings_path = args.settings_path
    gmt_path = args.gmt_path
    scorer = Scorer()
    scorer.from_settings(settings_path)
    scorer.score_LPOCV(gmt_path=gmt_path)
    scorer.get_stats(gmt_path=gmt_path)
