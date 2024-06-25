import argparse

from aamutils.cmd.expanding import configure_parser as configure_expanding_parser


def setup_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="AAM Utils",
        description="A collection of atom-atom-mapping utility functions.",
    )
    subparsers = parser.add_subparsers(dest="command")

    configure_expanding_parser(subparsers)

    return parser
