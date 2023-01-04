import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import numpy as np
import os
import json
import argparse


def load_waterfall(x):
    tempfile = os.path.join("./temp", x.split("/")[-1])

    if os.path.exists(tempfile):
        try:
            with open(tempfile, 'r') as file_in:
                return int(file_in.readlines()[0])
        except:
            print("Could not read saved file...")

    summation = np.sum(np.fromfile(x, dtype=np.uint8), dtype=int)
    with open(tempfile, 'w') as file_out:
        file_out.write(str(summation))
    return summation


def main(flags):
    if not os.path.exists('./temp'):
        os.mkdir("./temp")

    if not os.path.isfile(flags.csv):
        print(f'could not find {flags.csv}')
        exit()

    train = dd.read_csv(flags.csv)
    train['waterfall_sum'] = train['waterfall_location'].apply(lambda x: load_waterfall(x), meta=('waterfall_sum', int))
    with ProgressBar():
        train = train.compute()

    mu = sum(train['waterfall_sum']) / (623 * 1542 * train.shape[0])
    sigma = sum(train['waterfall_sum']) / (623 * 1542 * train.shape[0] - 1) ** .5

    # Save Mu and Sigma
    with open(flags.stats, 'w') as file_out:
        json.dump({
            'mu': mu,
            'sigma': sigma
        }, file_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--csv', type=str,
                        default='obs_list.json.csv',
                        help='The CSV to process and extract stats for.')

    parser.add_argument('--stats', type=str,
                        default="./satnogs-data/stats.json",
                        help='The place to save the stats')

    parsed_flags, _ = parser.parse_known_args()

    main(parsed_flags)
