import argparse
from .score import Scorer


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
