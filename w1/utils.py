# TODO colourful prints
import os

COLOR_WARNING = "\x1b[0;30;43m"

def make_dirs(path):
    """
    check if dir exists, if not: creates it
    :param path: path to create
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Creating path {path}")


def print_colored(string: str, color_id):
    """
    prints a string colorized
    :param string: string to colorize
    :param color_id: reference https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal
    :return:
    """
    print(color_id + string + '\x1b[0m')


def colorize_string(string: str, color_id):
    """
    colorizes a string to use with "print" function
    :param string: string to colorize
    :param color_id: color_id reference: https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal
    :return: concatenated string with colorize format
    """
    return color_id + string + '\x1b[0m'
