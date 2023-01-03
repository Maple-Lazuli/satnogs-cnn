import pandas as pd
import numpy as np
import os
import json
import argparse

def load_waterfall(x):
    return np.fromfile(x, dtype=np.uint8).reshape(-1, 623)

def main(flags):

    if not os.path.isfile(flags.csv):
        print(f'could not find {flags.csv}')
        exit()

    train = pd.read_csv(flags.csv)
    train['loaded_waterfall'] = train['waterfall_location'].apply(lambda x: load_waterfall(x))
    train['waterfall_sum'] = train['loaded_waterfall'].apply(lambda x: np.sum(x))
    mu = sum(train['waterfall_sum']) / (623 * 1542 * train.shape[0])

    train['loaded_waterfall'] = train['loaded_waterfall'].apply(lambda x: (x - mu) ** 2)
    train['waterfall_sum'] = train['loaded_waterfall'].apply(lambda x: np.sum(x))
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