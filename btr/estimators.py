from mord import LogisticAT
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def get_estimator(settings):
    """Returns an estmator as specified in the settings dict."""
    name = settings["name"]
    params = settings["params"]
    if name in ["Ordinal", "mord.LogisticAT"]:
        return LogisticAT(**params)
    elif name in ["Multiclass_Linear",
                  "sklearn.linear_model.LogisticRegression"]:
        return LogisticRegression(**params)
    elif name in ["Multiclass_Nonlinear",
                  "sklearn.ensemble.RandomForestClassifier"]:
        return RandomForestClassifier(**params)
    else:
        raise NotImplementedError
