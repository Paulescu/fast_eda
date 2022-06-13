from collections import OrderedDict
from typing import List
from pdb import set_trace as stop

import pandas as pd


# raw features we must have in the input data
RAW_FEATURES = {
    'gender': 'string',
    'SeniorCitizen': 'string',
    'Partner': 'string',
    'Dependents': 'string',
    'tenure': 'numerical',
    'PhoneService': 'string',
    'MultipleLines': 'string',
    'InternetService': 'string',
    'PaymentMethod': 'string',
    'MonthlyCharges': 'numeric',
    'TotalCharges': 'numeric',
    'date': 'datetime',
    'activity': 'string,'
}

# features we generate from the raw ones
ENGINEERED_FEATURES = [
    'dayOfWeek',
    'month',
    'dayOfMonth',
    'hour',
]

FEATURES_TO_MODEL = OrderedDict({
    # raw features
    'gender': 'string',
    'SeniorCitizen': 'string',
    'Partner': 'string',
    'Dependents': 'string',
    'tenure': 'string',
    'PhoneService': 'string',
    'MultipleLines': 'string',
    'InternetService': 'string',
    'PaymentMethod': 'string',
    'MonthlyCharges': 'numeric',
    'TotalCharges': 'numeric',
    'activity': 'string',

    # engineered features
    'dayOfWeek': 'numeric',
    'month': 'numeric',
    'dayOfMonth': 'numeric',
    'hour': 'numeric',
})


def enforce_feature_types(data_: pd.DataFrame) -> pd.DataFrame:
    """"""
    data = data_.copy()

    for feature, type_ in RAW_FEATURES.items():

        if type_ == 'numeric':
            data[feature] = pd.to_numeric(data[feature], errors='coerce')
        elif type_ == 'datetime':
            data[feature] = pd.to_datetime(data[feature], errors='coerce')
        elif type_ == 'string':
            data[feature] = data[feature].astype('str')

    return data


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """"""
    data = enforce_feature_types(data)

    try:
        data = data[list(RAW_FEATURES.keys())]
    except:
        print('Missing raw features in the input!')

    # add engineered features
    for feature in ENGINEERED_FEATURES:
        func = eval('add_feature_' + feature)
        data = func(data)

    # keep only features in FEATURES_TO_MODEL
    data = data[list(FEATURES_TO_MODEL.keys())]

    return data


def add_feature_month(data: pd.DataFrame) -> pd.DataFrame:
    """"""
    data['month'] = [t.month for t in data['date']]
    return data

def add_feature_dayOfMonth(data: pd.DataFrame) -> pd.DataFrame:
    """"""
    data['dayOfMonth'] = [t.day for t in data['date']]
    return data

def add_feature_dayOfWeek(data: pd.DataFrame) -> pd.DataFrame:
    """"""
    data['dayOfWeek'] = [t.dayofweek for t in data['date']]
    return data

def add_feature_hour(data: pd.DataFrame) -> pd.DataFrame:
    """"""
    data['hour'] = [t.hour for t in data['date']]
    return data