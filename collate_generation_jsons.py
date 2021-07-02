#! /usr/bin/env python
"""
Collate and merge the evaluation JSONs produced by the model.

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import collections
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

    dialog_id_map = collections.defaultdict(list)
    for ii in all_dialogs:
        # If map already exists, skip it.
        if ii["dialog_id"] not in dialog_id_map:
            dialog_id_map[ii["dialog_id"]].append(ii)
    all_dialogs = [ii[0] for ii in dialog_id_map.values()]

    # If predicting belief states, unflatten and save.
    if args["predict_belief_states"]:
        dialogs_reformat = []
        num_instances = 0
        for dialog_datum in all_dialogs:
            new_dialog = {
                "dialogue_idx": dialog_datum["dialog_id"], "dialogue": []
            }
            for pred_datum in dialog_datum["predictions"]:
                num_instances += 1
                output = parse_flattened_result(pred_datum["response"])
                new_dialog["dialogue"].append(
                    {
                        "turn_idx": pred_datum["turn_id"],
                        "belief_state": output,
                    }
                )
            dialogs_reformat.append(new_dialog)
        all_dialogs = {"dialogue_data": dialogs_reformat}
        print("# instances (Belief States): {}".format(num_instances))
    else:
        print("# instances: {}".format(len(all_dialogs)))

    print("Saving: {}".format(args["collated_save_path"]))
    with open(args["collated_save_path"], "w") as file_id:
        json.dump(all_dialogs, file_id)


def parse_flattened_result(to_parse):
    """
    Parse out the belief state from the raw text.
    Return an empty list if the belief state can't be parsed

    Input:
    - A single <str> of flattened result
      e.g. 'User: Show me something else => Belief State : DA:REQUEST ...'

    Output:
    - Parsed result in a JSON format, where the format is:
        [
            {
                'act': <str>  # e.g. 'DA:REQUEST',
                'slots': [
                    <str> slot_name,
                    <str> slot_value
                ]
            }, ...  # End of a frame
        ]  # End of a dialog
    """
    dialog_act_regex = re.compile(r'([\w:?.?]*)  *\[([^\]]*)\] *\(([^\]]*)\) *\<([^\]]*)\>')
    slot_regex = re.compile(r'([A-Za-z0-9_.-:]*)  *= ([^,]*)')
    request_regex = re.compile(r'([A-Za-z0-9_.-:]+)')
    object_regex = re.compile(r'([A-Za-z0-9]+)')

    belief = []
    # Parse
    to_parse = to_parse.strip()
    for dialog_act in dialog_act_regex.finditer(to_parse):
        d = {
            'act': dialog_act.group(1),
            'slots': [],
            'request_slots': [],
            'objects': []
        }

        for slot in slot_regex.finditer(dialog_act.group(2)):
            d['slots'].append(
                [
                    slot.group(1).strip(),
                    slot.group(2).strip()
                ]
            )

        for request_slot in request_regex.finditer(dialog_act.group(3)):
            d['request_slots'].append(request_slot.group(1).strip())

        for object_id in object_regex.finditer(dialog_act.group(4)):
            d['objects'].append(object_id.group(1).strip())

        if d != {}:
            belief.append(d)
    return belief


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
    parser.add_argument(
        "--predict_belief_states",
        default=False,
        action="store_true",
        help="Indicate if predicted responses are belief states",
    )

    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)
