import json
from .processing_schemes import LPOCV


class Loader(object):
    def __init__(self, settings_path=None):
        self.s = None
        if settings_path:
            with open(settings_path) as f:
                self.s = json.load(f)

    def processor_from_settings(self):
        settings = self.s
        name = settings["processing_scheme"]["name"]
        if name == 'LPOCV':
            return LPOCV(settings=settings)
        else:
            raise NotImplementedError
