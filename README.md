# AiCrowd-InsuranceGame

Tamas Papp's, My attempt at the AiCrowd Insurance market simulation competition.

We no longer have access to the data so this is just left here for legacy.

## Python env

run the following
`conda create --name insurance`
`conda activate insurance`
`pip install python/requirements.txt`

If you want to use some jupyter notebooks 
`pip install ipykernel`
`python -m ipykernel install --user --name insurance`


## Make Submission

- `export Insurance=Path to this repository`
- Run `train.py` you don't need to specify the data file if Insurance environment variable is defined
- Test with `source test.sh`
- Zip the python/ folder into submission.zip by running source `make_submission.sh`. This zip file can be uploaded as a submission


## Quick Guide to the Code in the python directory

### `model.py` 

Template required for the competition. All feature engineering is done here (we didn't do much on that front!)
Think the only slightly creative one is if they have purchased a new car or not.
For the rest of the functions we do all the coding in `model_classes.py`

### `model_classes.py`

The implementation works with config files stored in ensemble models. The dictionary taken from the config files, direct the initiate function to the relevant method.

Each class represents an approach we aspired to use. In the end we just used the LGBM class with tweedie loss. This was what ended up being best in our simulations. The config files in the ensemble_configs directory summarise the rest of our attempted approaches.

Pricing strategy is defined in general_predictor.predict_price. It implements a polynomial pricing. We used a quadratic for our final solution (0.01+ 1.0482E(claim)+7e-4 square(E(claim)) as seen in ensemble_models/tweedie_lgbm2.json.

There is an NGBoost implementation but this doesn't work due to a buggy one hot encoder hence we didn't use it in the end.
We were going to try a strategy with price = E[claims]+multipler*SD(claims)

