import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description="Pipeline Options",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data_path", type=str, default="./dataset/example.csv")

    parser.add_argument("--int", type=int, default=64)
    parser.add_argument("--redirector", action='store_false')

    return parser
