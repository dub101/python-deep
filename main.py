#!/usr/bin/env python
import argparse
import logging
import time

from vision_networks import MultilayerPerceptron


def set_logger():
    """
    Description:
        Configures logger for main script
    Parameters:
        None
    Return:
        Logger
    """

    logger = logging.getLogger("main logger")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(
        logging.Formatter(
            "{asctime} - {name} - {levelname}\n{message}",
            style="{"
    ))
    logger.addHandler(ch)

    return logger


def get_args():
    """
    Description:
        Collects command line arguments for main script
    Parameters:
        None
    Return:
        Arguments
    """

    parser = argparse.ArgumentParser(
        prog="Neural networks exploration",
        description="Program to explore, build, train and make inferences with neural networks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-n",
        "--name",
        default="multilayer_perceptron",
        type=str,
        choices=[
            "multilayer_perceptron",
        ],
        help="Choose the neural network name",
        metavar="name"
    )

    parser.add_argument(
        "-a",
        "--activity",
        default="train",
        type=str,
        choices=[
            "training",
            "inference"
        ],
        help="Choose the activity to be performed",
        metavar="activity"
    )

    return parser.parse_args()


if __name__ == "__main__":
    logger = set_logger()
    args = get_args()

    logger.info("Starting main script for neural networks exploration")

    if args.name == "multilayer_perceptron":
        nn = MultilayerPerceptron.MultilayerPerceptron()
        nn.train()



