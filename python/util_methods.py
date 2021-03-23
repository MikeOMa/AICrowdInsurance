import pandas as pd
from model import preprocess_data
import os

try:
    input_dataset = os.environ["Insurance"] + "/data/training.csv"
except:
    input_dataset = "data/training.csv"  # The default value.


def read_data(input_dataset=input_dataset):
    data = pd.read_csv(input_dataset)
    return data


def read_and_process(input_dataset=input_dataset):
    data = read_data(input_dataset)
    Xproc = preprocess_data(data.drop(columns=["claim_amount"]))
    Xproc["id_policy"] = data["id_policy"].copy()
    Xproc["claim_amount"] = data["claim_amount"].copy()
    Xproc["year"] = data["year"].copy()
    return Xproc


def get_xygroup(dataset):
    """

    Args:
        dataset: pd.Dataframe, created by read_and_process

    Returns:

        X,y and groups for kfold or machine learning
    """
    X = dataset.drop(columns=["claim_amount", "id_policy", "year"])
    y = dataset["claim_amount"]
    group = dataset["id_policy"]
    return X, y, group
