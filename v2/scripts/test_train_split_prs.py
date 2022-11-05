import os
from pathlib import Path
import argparse
import pandas as pd

# from azureml.core import Run, Datastore, Experiment
from azureml_user.parallel_run import EntryScript


def my_parse_args():
    parser = argparse.ArgumentParser("Test")

    parser.add_argument("--output_dir_train", type=str)
    parser.add_argument("--output_dir_test", type=str)

    args, _ = parser.parse_known_args()
    return args


def init():
    global train_output
    global test_output
    global logger

    # run = Run.get_context()
    # ws = run.experiment.workspace
    # dstore = Datastore.get_default(ws)

    args = my_parse_args()
    train_output = Path(args.output_dir_train)
    test_output = Path(args.output_dir_test)

    entry_script = EntryScript()
    # output_dir = Path(entry_script.output_dir)
    train_output.mkdir(parents=True, exist_ok=True)
    test_output.mkdir(parents=True, exist_ok=True)

    logger = entry_script.logger
    logger.info(f"$$$ train_output_folder => {train_output}")
    logger.info(f"$$$ test_output_folder => {test_output}")
    # logger.info(f"--- entry script output dir => {output_dir}")


def run(mini_batch):
    resultList = []

    for f in mini_batch:
        df = pd.read_csv(f)
        test_split_idx = int(df.shape[0] * 0.8)
        train_df = df.iloc[:test_split_idx]
        test_df = df.iloc[test_split_idx:]

        logger.info(f"### input_file_name => {f}")
        train_file = train_output / f"{f.split('.')[0].split('/')[-1]}.parquet"
        test_file = test_output / f"{f.split('.')[0].split('/')[-1]}.parquet"

        train_df.to_parquet(train_file, compression=None)
        test_df.to_parquet(test_file, compression=None)

        resultList.append("{}: {}".format(os.path.basename(f), "done"))
    return resultList
