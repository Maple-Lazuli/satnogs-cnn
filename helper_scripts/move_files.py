import argparse
from dask.diagnostics import ProgressBar
import dask.dataframe as dd
import os
import shutil


def move_file(source_path, destination):
    file_name = source_path.split("/")[-1]
    parent_dir = source_path.split("/")[-2]

    destination_abs = os.path.abspath(destination)
    new_parent_dir = os.path.join(destination_abs, parent_dir)
    new_source_path = os.path.join(new_parent_dir, file_name)

    if not os.path.exists(new_parent_dir):
        os.mkdir(new_parent_dir)

    if os.path.exists(new_source_path):
        if os.path.getsize(source_path) != os.path.getsize(new_source_path):
            shutil.copy2(source_path, new_source_path)
    else:
        shutil.copy2(source_path, new_source_path)

    return new_source_path


def main(flags):
    destination = flags.dest_dir

    ddf = dd.read_csv(flags.source_csv)

    ddf['waterfall_location'] = ddf['waterfall_location'].apply(lambda x: move_file(x, destination),
                                                                meta=('waterfall_location', str))

    with ProgressBar():
        ddf = ddf.compute()

    ddf.to_csv(flags.new_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_csv', type=str,
                        default='train.csv',
                        help='The CSV to process')

    parser.add_argument('--dest_dir', type=str,
                        default='./dest/',
                        help='The location to move the files to')

    parser.add_argument('--new_csv', type=str,
                        default='new_train.csv',
                        help='The CSV with the updated paths')

    parsed_flags, _ = parser.parse_known_args()

    main(parsed_flags)
