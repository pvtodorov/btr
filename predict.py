import argparse
from processing_schemes import Loader
from gmt import GMT

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("settings_path", help="settings as JSON")
    parser.add_argument("-i", "--iterations", help="iterations for the script",
                        required=False, default="1")
    parser.add_argument("-g", "--gmt_path",
                        help="path to file or folder of txts", required=False)
    argument = parser.parse_args()
    args = parser.parse_args()
    settings_path = args.settings_path
    iterations = int(args.iterations)
    gmt_path = args.gmt_path
    gmt = None
    if gmt_path:
        gmt = GMT(gmt_path)
    proc = Loader.processor_from_settings(settings_path=settings_path)
    proc.from_settings()
    if gmt:
        proc.predict_gmt(gmt)
    else:
        proc.predict_background()
    proc.save_results(gmt)
