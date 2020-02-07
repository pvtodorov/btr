from mord import LogisticAT
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from xgboost import XGBClassifier
import numpy as np



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
    elif name in ["xgboost.XGBClassifier"]:
        return XGBClassifier(**params)
    elif name in ["sklearn.linear_model.RidgeClassifier"]:
        return RidgeClassifierBTR(**params)
    else:
        raise NotImplementedError

class RidgeClassifierBTR(RidgeClassifier):
    def __init__(self, *args, **kwargs):
        RidgeClassifier.__init__(self, *args, **kwargs)
    def predict_proba(self, X_test):
        """Returns the distance from the hyperplane of the RidgeClassifer.
        Not a probability, per se, but values are still rankable for LPOCV.

        Probability for a pair looks like this:
        array([[0.56079167, 0.43920833],
               [0.562586  , 0.437414  ]])

        The decision function produces distance from hyperplane for each sample.
        predict_proba usually produces one set of probabilities per sample
        [p_class_A, p_class_B] where p_class_A = 1-p_class_B

        To produce something similar to this, negate the distances.
        Append negated values, and return transposed array.
        [dist_hyperplane, -dist_hyperplane]
        """
        probas = self.decision_function(X_test)
        probas = np.transpose(np.array([probas, -probas]))
        return probas