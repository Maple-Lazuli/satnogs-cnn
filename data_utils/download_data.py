import json
import satnogs_webscraper as sw
import pandas as pd
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


if __name__ == "__main__":
    good = sw.Scraper(save_name="good", good=True, bad=False, waterfall=1, artifacts=1, list_page_limit=5)
    bad = sw.Scraper(save_name="bad", good=False, bad=True, waterfall=1, artifacts=2, list_page_limit=5)
    good_df = good.scrape()
    bad_df = bad.scrape()
    combined_df = pd.concat([good_df, bad_df])
    drop = combined_df['Downloads'].apply(lambda x: filter_observation(x))
    combined_df = combined_df[drop]
    combined_df = combined_df.sample(frac=1)

    combined_df['waterfall_location'] = combined_df['Downloads'].apply(lambda x: get_waterfall(x))
    combined_df['status'] = combined_df['Status'].apply(lambda x: get_target(x))

    combined_len = int(combined_df.shape[0] * .8)
    train = combined_df[:combined_len]
    left = combined_df[combined_len:]

    left_len = int(left.shape[0] * .5)
    val = left[left_len:]
    test = left[:left_len]

    train.to_csv("./satnogs-data/train.csv", index=False)
    val.to_csv("./satnogs-data/val.csv", index=False)
    test.to_csv("./satnogs-data/test.csv", index=False)

    # Calculate Mu and Sigma

    train['loaded_waterfall'] = train['Downloads'].apply(lambda x: load_waterfall(x))
    train['waterfall_sum'] = train['loaded_waterfall'].apply(lambda x: np.sum(x))
    mu = sum(train['waterfall_sum']) / (623 * 1542 * train.shape[0])

    train['loaded_waterfall'] = train['loaded_waterfall'].apply(lambda x: (x - mu) ** 2)
    train['waterfall_sum'] = train['loaded_waterfall'].apply(lambda x: np.sum(x))
    sigma = sum(train['waterfall_sum']) / (623 * 1542 * train.shape[0] - 1) ** .5

    # Save Mu and Sigma
    with open("./satnogs-data/stats.json", 'w') as file_out:
        json.dump({
            'mu': mu,
            'sigma': sigma
        }, file_out)
