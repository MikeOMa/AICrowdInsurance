"""In this module, we ask you to define your pricing model, in Python."""

####
# Import modules
####
# Don't forget to add them to requirements.txt before submitting.
# NOTE THAT ANY TENSORFLOW VERSION HAS TO BE LOWER THAN 2.4. So 2.3XXX would work.

import json
import pickle
from model_classes import initiate


def preprocess_data(X_raw, inplace=False):
    """Convert 'object' columns to 'categorical' columns, for use by LightGBM.

    Parameters
    ----------
    X_raw : Pandas dataframe, where the categorical columns have object type.

    Returns
    ---------
    X_raw : Pandas dataframe, with columns of object type converted to categorical type.
    inplace:  bool, returns a copy if False, processing the df inplace if True
    """

    # We copy X_raw incase we need it later.
    # This is because we might need id_policy for the group CV but we don't
    # want it for prediction

    if inplace:
        X_new = X_raw
    else:
        X_new = X_raw.copy()

    years = X_new["year"].to_list()
    if not all(years[i] <= years[i + 1] for i in range(len(years) - 1)):
        print("Years column is not sorted, code will not work as intended.")

    # The following works if the df is sorted in terms of years.
    # It will lag the dataset within groups and put everything in the right place
    # Could do 1-4 below but i found it drops the score a bit.
    for i in range(1, 3):
        X_new["pol_no_claims_discount" + str(i) + "y"] = X_new.groupby("id_policy")[
            "pol_no_claims_discount"
        ].shift(i)
    X_new["pol_no_claims_delta"] = X_new.groupby("id_policy")[
        "pol_no_claims_discount"
    ].diff()

    X_new["vh_make_model_ly"] = X_new.groupby("id_policy")["vh_make_model"].shift(1)
    # if their new car is the same
    X_new["new_car"] = X_new["vh_make_model"] != X_new["vh_make_model_ly"]
    X_new["new_car"] = X_new["new_car"].astype("float32")

    # Extract the categorical columns to a list
    categorical_columns = list(X_new.select_dtypes(include="object").columns.values)

    print("Categorical " + "#" * 5)
    print(categorical_columns)

    # Convert categorical columns to correct type
    for column in categorical_columns:
        X_new[column] = X_new[column].astype("category")

    drop_columns = ["id_policy", "year", "vh_make_model_ly"]
    drop_columns.append("vh_make_model")
    print("DROPPING " + "#" * 5)
    print(drop_columns)

    X_cols = [i for i in X_new.columns if i not in drop_columns]
    X_new = X_new[X_cols]
    return X_new


####
# Necessary functions
####
def fit_model(X_raw, y_raw, config="best_config.json"):
    """Model training function: given training data (X_raw, y_raw), train this pricing model.

    Parameters
    ----------
    X_raw : Pandas dataframe, with the columns described in the data dictionary.
        Each row is a different contract. This data has not been processed.
    y_raw : a Numpy array, with the value of the claims, in the same order as contracts in X_raw.
        A one dimensional array, with values either 0 (most entries) or >0.
    config : a json filename, should direct to a configuration file that works for this project.
            model_type key, value within the file is mandatory
    Returns
    -------
    self: this instance of the fitted model. This can be anything, as long as it is compatible
        with your prediction methods.
    """

    # Preprocess the data
    pricing_params = dict(json.load(open(config, "r")))

    # Copy to make sure this group is not modified
    # Do not shuffle the X data without updating group if using Cross Validation.

    pricing_params["groups"] = X_raw["id_policy"].values.copy()

    # Currently this is the only feature we're dropping
    # If we decided to do feature selection do something more advanced with the config file
    X_processed = preprocess_data(X_raw)

    # Create dataset LightGBM can work with, categorical data dealt with automatically
    model = initiate(pricing_params)
    model.fit(X_processed, y_raw)
    return model


def predict_expected_claim(model, X_raw):
    """Model prediction function: predicts the expected claim based on the pricing model.

    This functions estimates the expected claim made by a contract (typically, as the product
    of the probability of having a claim multiplied by the expected cost of a claim if it occurs),
    for each contract in the dataset X_raw.

    This is the function used in the RMSE leaderboard, and hence the output should be as close
    as possible to the expected cost of a contract.

    Parameters
    ----------
    model: a Python object that describes your model. This can be anything, as long
        as it is consistent with what `fit` outpurs.
    X_raw : Pandas dataframe, with the columns described in the data dictionary.
        Each row is a different contract. This data has not been processed.

    Returns
    -------
    avg_claims: a one-dimensional Numpy array of the same length as X_raw, with one
        expected claim per contract (in same order). These expected claims must be POSITIVE (>0).
    """

    # Estimate the expected claim of every contract.
    X_processed = preprocess_data(X_raw)
    avg_claims = model.predict(X_processed)

    return avg_claims


def predict_premium(model, X_raw):
    """Model prediction function: predicts premiums based on the pricing model.

    This function outputs the prices that will be offered to the contracts in X_raw.
    premium will typically depend on the expected claim predicted in
    predict_expected_claim, and will add some pricing strategy on top.

    This is the function used in the expected profit leaderboard. Prices output here will
    be used in competition with other models, so feel free to use a pricing strategy.

    Parameters
    ----------
    model: a Python object that describes your model. This can be anything, as long
        as it is consistent with what `fit` outputs.
    X_raw : Pandas dataframe, with the columns described in the data dictionary.
        Each row is a different contract. This data has not been processed.

    Returns
    -------
    prices: a one-dimensional Numpy array of the same length as X_raw, with one
        price per contract (in same order). These prices must be POSITIVE (>0).
    """

    # Return a price for everyone.
    X = preprocess_data(X_raw)
    prices = model.predict_price(X)

    return prices


def save_model(model):
    """Saves this trained model to a file.

    This is used to save the model after training, so that it can be used for prediction later.

    Do not touch this unless necessary (if you need specific features). If you do, do not
     forget to update the load_model method to be compatible.

    Parameters
    ----------
    model: a Python object that describes your model. This can be anything, as long
        as it is consistent with what `fit` outpurs."""

    model.save()


def load_model():
    """Load a saved trained model from the file.

    This is called by the server to evaluate your submission on hidden data.
    Only modify this *if* you modified save_model."""
    pricing_params = dict(json.load(open("best_config.json")))
    fname = pricing_params["fname"] + ".p"
    with open(fname, "rb") as target:
        trained_model = pickle.load(target)
    return trained_model
