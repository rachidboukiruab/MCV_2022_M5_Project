# TODO colourful prints
import os


def make_dirs(path):
    """
    check if dir exists, if not: creates it
    :param path: path to create
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Creating path {path}")