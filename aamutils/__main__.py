import logging

from aamutils.cmd import setup_argparser

if __name__ == "__main__":
    logger = logging.getLogger("aamutils")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s %(levelname)s] %(message)s", datefmt="%y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(ch)

    parser = setup_argparser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception:
        parser.print_help()
