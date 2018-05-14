from mord import LogisticAT
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def get_estimator(settings):
    """Returns an estmator as specified in the settings dict."""
    name = settings["name"]
    params = settings["params"]
    if name == "Ordinal":
        return LogisticAT(**params)
    elif name == "Multiclass_Linear":
        return LogisticRegression(**params)
    elif name == "Multiclass_Nonlinear":
        return RandomForestClassifier(**params)
    else:
        raise NotImplementedError
