from collections import OrderedDict
from typing import List
from pdb import set_trace as stop

import pandas as pd


# raw features we must have in the input data
RAW_FEATURES = {
    'churn_predictor': {
        'bidPrice': 'numeric',
        'zipcode': 'string',
        'state': 'string',
        'age': 'numeric',
        'source': 'string',
        'subid1': 'string',
        'bidDate': 'datetime',
    },
}

# features we generate from the raw ones
ENGINEERED_FEATURES = {
    'churn_predictor': [
        'bidDateDayOfWeek',
        'bidDateMonth',
        'bidDateDayOfMonth',
        'bidDateHour',
    ],
}

FEATURES_TO_MODEL = {
    'churn_predictor': OrderedDict({
        # raw features
        # TODO

        # engineered features
        'bidDateDayOfWeek': 'numeric',
        'bidDateMonth': 'numeric',
        'bidDateDayOfMonth': 'numeric',
        'bidDateHour': 'numeric',
    }),
}


def enforce_feature_types(data_: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """"""
    data = data_.copy()

    for feature, type_ in RAW_FEATURES[model_name].items():

        if type_ == 'numeric':
            data[feature] = pd.to_numeric(data[feature], errors='coerce')
        elif type_ == 'datetime':
            data[feature] = pd.to_datetime(data[feature], errors='coerce')
        elif type_ == 'string':
            data[feature] = data[feature].astype('str')

    return data


def add_features(data: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """"""
    assert model_name in {'churn_predictor'}, 'Invalid model_name man!'

    data = enforce_feature_types(data, model_name)

    try:
        data = data[list(RAW_FEATURES[model_name].keys())]
    except:
        print('Missing raw features in the input!')

    # add engineered features
    for feature in ENGINEERED_FEATURES[model_name]:
        func = eval('add_feature_' + feature)
        data = func(data)

    # keep only features in FEATURES_TO_MODEL
    data = data[list(FEATURES_TO_MODEL[model_name].keys())]

    return data


def add_feature_bidDateMonth(data: pd.DataFrame) -> pd.DataFrame:
    """"""
    data['bidDateMonth'] = [t.month for t in data['bidDate']]
    return data

def add_feature_bidDateDayOfMonth(data: pd.DataFrame) -> pd.DataFrame:
    """"""
    data['bidDateDayOfMonth'] = [t.day for t in data['bidDate']]
    return data

def add_feature_bidDateDayOfWeek(data: pd.DataFrame) -> pd.DataFrame:
    """"""
    data['bidDateDayOfWeek'] = [t.dayofweek for t in data['bidDate']]
    return data

def add_feature_bidDateHour(data: pd.DataFrame) -> pd.DataFrame:
    """"""
    data['bidDateHour'] = [t.hour for t in data['bidDate']]
    return data

def add_feature_osGroup(data: pd.DataFrame) -> pd.DataFrame:
    """"""
    def os2group(x_: str) -> str:

        x = str(x_)

        if 'android' in x.lower():
            return 'Android'
        elif 'windows' in x.lower():
            return 'Windows'
        elif 'mac' in x.lower():
            return 'Mac'
        elif 'ios' in x.lower():
            return 'iOS'
        elif 'chrome' in x.lower():
            return 'Chrome'
        else:
            return 'Other'

    data['osGroup'] = data['os'].apply(os2group)
    return data

def add_feature_browserGroup(data: pd.DataFrame) -> pd.DataFrame:
    """"""
    def browser2group(x_: str) -> str:

        x = str(x_)

        if 'facebook' in x.lower():
            return 'Facebook'
        elif 'chrome mobile' in x.lower():
            return 'Chrome Mobile'
        elif 'mobile safari' in x.lower():
            return 'Mobile Safari'
        elif 'samsung' in x.lower():
            return 'Samsung'
        elif 'chrome' in x.lower():
            return 'Chrome'
        elif 'safari' in x.lower():
            return 'Safari'
        elif 'edge' in x.lower():
            return 'Edge'
        elif 'firefox' in x.lower():
            return 'Firefox'
        elif 'instagram' in x.lower():
            return 'Instagram'
        else:
            return 'Other'

    data['browserGroup'] = data['browser'].apply(browser2group)
    return data


def reduce_cardinality_categorical_feature(
    data_: pd.Series,
    threshold: float
) -> pd.Series:
    """"""
    data = data_.copy()

    # make sure missing values are kept later on, by using a new identifier 'null'
    data.fillna('null', inplace=True)

    df = data.value_counts(normalize=True).to_frame('freq')
    df['cum'] = df['freq'].cumsum()

    # keep top X categories
    df = df[df.cum <= threshold]
    top_categories = df.index.values

    # collapse names
    return data.apply(lambda x: x if x in top_categories else 'other')


def add_feature_topCity(data: pd.DataFrame) -> pd.DataFrame:
    """"""
    threshold = 0.80
    data['topCity'] = reduce_cardinality_categorical_feature(data['city'], threshold=threshold)
    return data


def add_feature_topZipcode(data: pd.DataFrame) -> pd.DataFrame:
    """"""
    threshold = 0.80
    data['topZipcode'] = reduce_cardinality_categorical_feature(data['zipcode'], threshold=threshold)
    return data


def add_feature_topSubid1(data: pd.DataFrame) -> pd.DataFrame:
    """"""
    threshold = 0.80
    data['topSubid1'] = reduce_cardinality_categorical_feature(data['subid1'], threshold=threshold)
    return data


def add_feature_topSubid2(data: pd.DataFrame) -> pd.DataFrame:
    """"""
    threshold = 0.50
    data['topSubid2'] = reduce_cardinality_categorical_feature(data['subid2'], threshold=threshold)
    return data