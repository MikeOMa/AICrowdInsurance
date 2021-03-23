import lightgbm
import numpy as np
import pickle
import json

### For NGBoost freq-sev model
import pandas as pd

from lightgbm import LGBMClassifier, LGBMRegressor
from ngboost import NGBoost, NGBRegressor
from ngboost.scores import LogScore
from ngboost.distns import Bernoulli, LogNormal, Normal

###

from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import OneHotEncoder


####
# Our custom functions
####
def initiate(kw_dict):
    """Selects the model class from the model_implementations list above.
    Input:
        kw_dict all neccessary kwargs for the model in the future.
    Returns:
         an instance of the class refereed to in kw_dict['model_type']
        With all other parameters stored.
    """
    model_type = kw_dict["model_type"]
    model_implementations = {
        "lightgbm": LGBM,
        "lightgbm_freq_sev": LGBM_freq_sev,
        "ensemble": ensemble,
        "NGB_freq_sev": NGB_freq_sev,
        "ngboost": NGB,
    }
    return model_implementations[model_type](**kw_dict)


class general_predictor:
    """A template class, it initiates the predictor,
        This is only used to subclass so we can inherit useful parameters like
        pricing_mult and groups.

    Parameters
    ----------
    **kwargs : a configuration for the object

    """

    # Initializer
    def __init__(self, **kwargs):
        """
        Stores whatever is in kwargs as self.key = value
        """

        for l in kwargs.keys():
            setattr(self, l, kwargs[l])

        # initiate the groups parameter
        if not hasattr(self, "kcv"):
            self.kcv = 5

        # Use this as default price multiplier, if no custom method given.
        if not hasattr(self, "pricing_mult"):
            self.pricing_mult = [0, 1.3]

        if not hasattr(self, "drop_vars"):
            self.drop_vars = ["vh_make_model"]

        # set this to policy id if you want grouped cross validation
        if not hasattr(self, "groups"):
            self.groups = None

    def get_cv_folds(self, X, y):
        if self.groups is None:
            splitter = KFold(self.kcv)
        else:
            splitter = GroupKFold(self.kcv)

        ## OVERRIDE THE ABOVE IF CV_TYPE IS DEFINED
        if hasattr(self, "cv_type"):
            if self.cv_type == "normal":
                splitter = KFold(self.kcv)

        folds = splitter.split(X, y, groups=self.groups)
        return folds

    def get_validation_set(self, X, y):
        folds = self.get_cv_folds(X, y)
        # Folds[0] is a iterable of length two
        return list(folds)[0]

    def fit(self, X, y):
        """
        Just a place holder
        """
        pass

    def predict(self, X):
        """
        Placeholder, should be overridden
        """
        if hasattr(self, "model"):
            # if the attribute model is defined we use model.predict by default
            ret = self.model.predict(X)
        else:
            ret = np.array([0] * X.shape[0])
        return ret

    def predict_price(self, X):
        """
        Parameters
        ----------
        X: pd.DataFrame of X variables

        Returns
        -------
        prices: np.ndarray

        Use self.price to transform predictions polynomially: self.price = [Intercept, linear coeff., quadratic coeff., ...].
        By default, the prices = 1.3 * preds.
        """
        preds = self.predict(X)
        prices = np.zeros(shape=len(preds))

        pows = range(len(self.pricing_mult))

        print(pows)
        for j in pows:
            prices = prices + self.pricing_mult[j] * preds ** j

        return prices

    def save(self):
        fname = self.fname + ".p"
        print("saving model " + fname)
        pickle.dump(self, open(fname, "wb"))


class LGBM(general_predictor):
    def fit(self, X, y):

        # Load in LighGBM parameter dictionary from a JSON file
        lgb_params = self.params

        # Group K-fold CV splitter from sklearn
        # Should be better as the leaderboard is based on leaving out full policy_ids.
        folds = self.get_cv_folds(X, y)
        lgb_train = lightgbm.Dataset(X, label=y)

        # Do rich man's early stopping, via cross-validation
        model_cv = lightgbm.cv(
            params=lgb_params,
            train_set=lgb_train,
            num_boost_round=4000,
            early_stopping_rounds=500,
            folds=folds,
        )

        # Find stopping iteration
        metric = self.params["metric"]
        metric_evals = model_cv[metric + "-mean"]
        best_iter = np.argmin(metric_evals)
        print(metric)
        print(metric_evals[best_iter])

        # Do actual training
        self.model = lightgbm.train(
            params=lgb_params, train_set=lgb_train, num_boost_round=best_iter
        )


class LGBM_freq_sev(general_predictor):
    def fit(self, X, y):
        mask_claims = y > 0
        binary = mask_claims.astype("int")

        self.models = []
        for k in ["params_freq", "params_sev"]:
            new_dict = getattr(self, k)
            self.models.append(initiate(new_dict))

        self.models[0].fit(X=X, y=binary)
        claims_X_data, claims_Y_data = X[mask_claims], y[mask_claims]
        self.models[1].fit(X=claims_X_data, y=claims_Y_data)

    def predict(self, X):
        freq = self.models[0].predict(X)
        sev = self.models[1].predict(X)
        return freq * sev


class ensemble(general_predictor):
    def fit(self, X, y):
        self.ensemble_dict = {"max": np.max, "mean": np.mean}
        # Select the ensembe method using the configuration
        self.ensemble_method = self.ensemble_dict[self.ens_method]

        # Initate all the models listed in files.
        self.models = []
        for k in self.files:
            par_dict = json.load(open(k))
            self.models.append(initiate(par_dict))

        # Fit each model
        self.N = len(self.models)
        print(self.N)
        for i in range(self.N):
            self.models[i].fit(X=X, y=y)

    def predict(self, X):
        preds = []
        for i in range(self.N):
            preds.append(self.models[i].predict(X=X).reshape(-1, 1))
        ensemble_preds = np.hstack(preds)
        final = self.ensemble_method(ensemble_preds, axis=1)
        fmean = {fname: np.mean(pred) for fname, pred in zip(self.files, preds)}
        print("mean prediction per model")
        print(fmean)
        return final

    def predict_price(self, X):
        preds = []
        for i in range(self.N):
            preds.append(self.models[i].predict_price(X=X).reshape(-1, 1))
        ensemble_preds = np.hstack(preds)
        fmean = {fname: np.mean(pred) for fname, pred in zip(self.files, preds)}
        print("mean price prediction per model")
        print(fmean)
        return self.ensemble_method(ensemble_preds, axis=1)


class NGB_freq_sev(general_predictor):
    def fit(self, X, y):
        mask_claims = y > 0
        binary = mask_claims.astype("int")

        ### Do some preprocessing
        X_OH = pd.get_dummies(X)  # One-hot encode
        X_OH.fillna(
            value=10 ** 6, inplace=True
        )  # HACK: Replace all missing values with a big number

        ### Create the models
        self.models = []

        # Set up frequency model
        freq = getattr(self, "params_freq")
        params_lightgbm = freq["params_lightgbm"]
        params_ngboost = freq["params_ngboost"]
        learner = LGBMRegressor(**params_lightgbm)

        params_ngboost["Base"] = learner
        params_ngboost["Dist"] = Bernoulli
        params_ngboost["Score"] = LogScore

        self.models.append(NGBoost(**params_ngboost))

        # Set up severity model
        sev = getattr(self, "params_sev")
        params_lightgbm = sev["params_lightgbm"]
        params_ngboost = sev["params_ngboost"]
        learner = LGBMRegressor(**params_lightgbm)

        params_ngboost["Base"] = learner
        params_ngboost["Dist"] = LogNormal
        params_ngboost["Score"] = LogScore

        self.models.append(NGBoost(**params_ngboost))

        claims_X_data, claims_Y_data = X_OH[mask_claims], y[mask_claims]
        claims_X_data = claims_X_data.to_numpy()
        self.models[1].fit(X=claims_X_data, Y=claims_Y_data)
        #### Fit the models on one-hot encoded data
        X = X_OH.to_numpy()
        self.models[0].fit(X=X, Y=binary)

    def predict(self, X):

        ### Can only predict on O-H encoded data
        X_OH = pd.get_dummies(X)  # One-hot encode
        X_OH.fillna(
            value=10 ** 6, inplace=True
        )  # HACK: Replace all missing values with a big number

        ### Frequency = prob of claim made
        freq_dists = self.models[0].pred_dist(X_OH.values)
        freq = freq_dists.params["p1"]

        #### Severity = average claim under log-normal distribution
        sev = self.models[1].predict(X_OH.values)
        return freq * sev


class NGB(general_predictor):
    def fit(self, X, y):
        ### Do some preprocessing
        params_ngboost = self.params_ngb
        params_lightgbm = self.params_lightgbm
        learner = LGBMRegressor(**params_lightgbm)
        self.OH = OneHotEncoder(sparse=False)
        X_OH = self.OH.fit_transform(X)
        print(self.OH.get_feature_names())
        ### Create the models
        dists = {"normal": Normal, "lognormal": LogNormal}
        dist = dists[self.dist]
        params_ngboost["Base"] = learner
        params_ngboost["Dist"] = dist
        params_ngboost["Score"] = LogScore

        self.model = NGBRegressor(**params_ngboost)

        train_idx, val_idx = self.get_validation_set(X, y)
        self.model.fit(
            X=X_OH[train_idx, :],
            Y=y[train_idx],
            X_val=X_OH[val_idx, :],
            Y_val=y[val_idx],
            early_stopping_rounds=200,
        )
        params_ngboost["n_estimators"] = self.model.best_val_loss_itr
        print(params_ngboost["n_estimators"])

        self.model = NGBoost(**params_ngboost)
        self.model.fit(X=X, Y=y.values)

    def predict(self, X):
        ### Can only predict on O-H encoded data

        X_OH = self.OH.transform(X)  # One-hot encode

        #### Severity = average claim under log-normal distribution
        return self.model.predict(X_OH)
