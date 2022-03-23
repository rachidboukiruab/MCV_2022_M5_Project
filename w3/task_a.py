from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def main(args: ArgumentParser):
    dataset_path = Path(args.dataset_path)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Inference tool for Task A",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path where to find Out-of-Context images"
    )
    args = parser.parse_args()

    main(args)
