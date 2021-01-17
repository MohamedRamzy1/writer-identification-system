from idlib.train import complete_train, sampled_train
from idlib.test import test

import argparse


def run(data_dir, mode):
    # call the appropriate function based on mode
    if mode == 'complete-train':
        complete_train(data_dir)
    elif mode == 'sampled-train':
        sampled_train(data_dir)
    elif mode == 'test':
        test(data_dir)
    else:
        raise Exception('incorrect mode!')


if __name__ == "__main__":
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-dir', '--data_dir', type=str,
        help='path to train/test data', default='data/'
    )
    argparser.add_argument(
        '-mode', '--process_mode', type=str,
        help='complete-train | sampled-train | test', default='test'
    )

    args = argparser.parse_args()

    # call main pipeline driver
    run(args.data_dir, args.process_mode)
