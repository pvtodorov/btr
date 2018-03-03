import argparse
from .loader import Loader
from .gmt import GMT


def main():
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
    for i in range(0, iterations):
        if gmt_path:
            gmt = GMT(gmt_path)
        loader = Loader(settings_path=settings_path)
        loader.get_processor_from_settings()
        loader.proc.from_settings()
        loader.proc.predict(gmt=gmt)
        loader.proc.save_results()
        loader.save_prediction_to_synapse()
