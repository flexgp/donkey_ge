import argparse
import yaml
from typing import Any, Dict

from heuristics import donkey_ge, donkey_ge_coev


__author__ = "Erik Hemberg"


"""
Main function for donkey_ge. Parses YML config file and call donkey_ge.
"""


def parse_arguments() -> Dict[str, Any]:
    """
    Parse command line arguments (`sys.argv`).

    :return: settings from configuration file and CLI arguments
    :rtype dict:
    """
    parser = argparse.ArgumentParser(description="Run donkey_ge")
    parser.add_argument(
        "-f",
        "--configuration_file",
        type=str,
        required=True,
        help="YAML configuration file. E.g. " "configurations/demo_ge.yml",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=".",
        help="Path to directory for output files. E.g. " "donkey_ge_output",
    )
    parser.add_argument("--coev", action="store_true", help="Coevolution")

    _args = parser.parse_args()

    # Read configuration file
    with open(_args.configuration_file, "r") as configuration_file:
        settings = yaml.load(configuration_file)

    # Set CLI arguments in settings
    settings["output_dir"] = _args.output_dir
    settings["coev"] = _args.coev

    return settings


def main() -> Dict[str, Any]:
    # Parse CLI arguments
    args = parse_arguments()
    if args["coev"]:
        donkey_ge_coev.run(args)
    else:
        donkey_ge.run(args)

    return args


if __name__ == "__main__":
    main()
