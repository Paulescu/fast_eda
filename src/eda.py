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

    target_data = data[target].tolist()

    # add features
    data = add_features(data, model_name=model_name)

    # keep target
    data[target] = target_data

    stop()
    data.to_csv(DATA_DIR / f'processed_{file_name}.csv')

    my_report = sv.analyze(data, target_feat=target)

    output_html = get_project_root() / 'viz' / f'sweetviz_{file_name}.html'
    my_report.show_html(output_html)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--file', dest='file', type=str)
    parser.add_argument('--target', dest='target', type=str)
    parser.add_argument('--model', dest='model', type=str)
    args = parser.parse_args()

    eda(file_name=args.file, target=args.target, model_name=args.model)