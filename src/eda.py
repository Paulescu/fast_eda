from argparse import ArgumentParser
from pathlib import Path
import os
from pdb import set_trace as stop

import sweetviz as sv
import pandas as pd

from src.features import add_features

ROOT_DIR = Path(__file__).parent.resolve().parent
DATA_DIR = ROOT_DIR / 'data'
VIZ_DIR = ROOT_DIR / 'viz'


def eda(
    file_name: str,
    target: str,
    model_name: str,
) -> None:

    # load data into pandas
    file_path = Path(DATA_DIR) / file_name
    data = pd.read_csv(file_path)

    # feature engineering
    target_data = data[target].tolist()
    data = add_features(data, model_name=model_name)
    data[target] = target_data

    # generate Sweetviz report
    my_report = sv.analyze(data, target_feat=target)
    output_html = VIZ_DIR / f'{file_name}.html'
    my_report.show_html(output_html)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--file', dest='file', type=str)
    parser.add_argument('--target', dest='target', type=str)
    parser.add_argument('--model', dest='model', type=str)
    args = parser.parse_args()

    eda(file_name=args.file, target=args.target, model_name=args.model)