#! /usr/bin/env python
"""
Collate and merge the evaluation JSONs produced by the model.

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import json
import re


def main(args):
    # Extract start step for each file.
    start_steps = {}
    for file_name in args["response_files"]:
        matches = re.findall(r"\S*start(\d*)\.\S*", file_name)
        assert len(matches) == 1, "Exactly one match!"
        start_steps[file_name] = int(matches[0])

    sorted_files = sorted(start_steps.items(), key=lambda x: x[1])
    all_dialogs = []
    for (file_name, start) in sorted_files:
        with open(file_name, "r") as file_id:
            current_dialogs = json.load(file_id)
            all_dialogs.extend(current_dialogs)

    print("# instances: {}".format(len(all_dialogs)))
    print("Saving: {}".format(args["collated_save_path"]))
    with open(args["collated_save_path"], "w") as file_id:
        json.dump(all_dialogs, file_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, fromfile_prefix_chars="@"
    )
    parser.add_argument(
        "--response_files",
        nargs="+",
        required=True,
        help="List of files to collate responses",
    )
    parser.add_argument(
        "--collated_save_path",
        required=True,
        help="JSON file to save all the collated responses"
    )

    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)
