import pandas as pd
import os
import json
import argparse


def fix_status(status):
    if type(status) == int:
        return status
    if status == 'Bad':
        return 0
    elif status == 'Good':
        return 1
    else:
        return -1


def fix_waterfall_path(download_str):
    dl_dict = json.loads(download_str.replace("'", '"'))
    waterfall_path = dl_dict['waterfall_hash_name']
    return os.path.abspath(waterfall_path)


def main(flags):
    if not os.path.isfile(flags.csv):
        print(f'could not find {flags.csv}')
        exit()

    df = pd.read_csv(flags.csv)

    df['status'] = df['Status'].apply(lambda x: fix_status(x))
    df['waterfall_location'] = df['Downloads'].apply(lambda x: fix_waterfall_path(x))

    df = df[['status', 'waterfall_location']]
    df = df.sample(frac=1)

    train_limit = int(400 * .8)
    left_over = int((df.shape[0] - train_limit) / 2)
    train_df = df[:train_limit]
    val_df = df[train_limit:train_limit + left_over]
    test_df = df[train_limit + left_over:]

    train_df.to_csv('train.csv', index=False)
    val_df.to_csv('val.csv', index=False)
    test_df.to_csv('test.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--csv', type=str,
                        default='obs_list.json.csv',
                        help='The CSV to process')

    parsed_flags, _ = parser.parse_known_args()

    main(parsed_flags)
