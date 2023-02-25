import json
import os.path
import argparse
import satnogs_webscraper as sw
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np


def filter_observation(dl_dict):
    if (dl_dict['waterfall_shape'][0] == 1542) and (dl_dict['waterfall_shape'][1] == 623):
        return True
    else:
        return False


def load_waterfall(x):
    return np.fromfile(x['waterfall_hash_name'], dtype=np.uint8).reshape(-1, 623)


def get_waterfall(x):
    return x['waterfall_hash_name']


def get_target(x):
    return 1 if x == "Good" else 0


def main(flags):
    norad = flags.norad
    page_limit = flags.page_limit
    save_dir = flags.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create scrapers then scrape
    good = sw.Scraper(save_name="good", good=True, bad=False, waterfall=1, artifacts=1, list_page_limit=page_limit,
                      norad=norad)
    bad = sw.Scraper(save_name="bad", good=False, bad=True, waterfall=1, artifacts=2, list_page_limit=page_limit,
                     norad=norad)
    good_df = good.scrape()
    bad_df = bad.scrape()

    # Combine the dataframes, shuffle, then rename columns.
    combined_df = pd.concat([good_df, bad_df])
    drop = combined_df['Downloads'].apply(lambda x: filter_observation(x))
    combined_df = combined_df[drop]
    combined_df = combined_df.sample(frac=1)
    combined_df['waterfall_location'] = combined_df['Downloads'].apply(lambda x: get_waterfall(x))
    combined_df['status'] = combined_df['Status'].apply(lambda x: get_target(x))

    # Create Train, Val, Test splits
    combined_len = int(combined_df.shape[0] * .8)
    train = combined_df[:combined_len]
    left = combined_df[combined_len:]
    left_len = int(left.shape[0] * .5)
    val = left[left_len:]
    test = left[:left_len]

    # Write CSVs to disk
    train.to_csv(f"{save_dir}/train.csv", index=False)
    train[train['status'] == 0].iloc[0:2].to_csv("./satnogs-data/train_1_example.csv", index=False)
    val.to_csv(f"{save_dir}/val.csv", index=False)
    test.to_csv(f"{save_dir}/test.csv", index=False)

    # Calculate Mu and Sigma
    train['loaded_waterfall'] = train['Downloads'].apply(lambda x: load_waterfall(x))
    train['waterfall_sum'] = train['loaded_waterfall'].apply(lambda x: np.sum(x))
    mu = sum(train['waterfall_sum']) / (623 * 1542 * train.shape[0])

    train['loaded_waterfall'] = train['loaded_waterfall'].apply(lambda x: (x - mu) ** 2)
    train['waterfall_sum'] = train['loaded_waterfall'].apply(lambda x: np.sum(x))

    sigma = sum(train['waterfall_sum']) / (623 * 1542 * train.shape[0] - 1) ** .5

    # Save Mu and Sigma
    with open(f"{save_dir}/stats.json", 'w') as file_out:
        json.dump({
            'mu': mu,
            'sigma': sigma
        }, file_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--norad', type=str,
                        default='44352',
                        help='The NORAD of the satellite to pull data for.')

    parser.add_argument('--page-limit', type=int,
                        default=5,
                        help='The number of pages to pull down.')

    parser.add_argument('--save-dir', type=str,
                        default="./satnogs-data/",
                        help='The directory to save the CSVs and statistics in.')

    parsed_flags, _ = parser.parse_known_args()

    main(parsed_flags)
