from . import build_parser, run


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
