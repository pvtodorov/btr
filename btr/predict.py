from .loader import Loader


def main():
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
        loader.get_processor_from_settings()
        loader.proc.from_settings()
        loader.proc.predict(gmt=gmt)
        loader.proc.save_results()
        loader.save_prediction_to_synapse()
