import argparse

import torch

############################################################################################################################################
# The ckp generated at stage 2 contains all the weights of both the teacher and the student. Now, we only keep the weights of the student. #
############################################################################################################################################


def clean_checkpoint(input_path, output_path):
    ckpt = torch.load(input_path, map_location="cpu")

    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("teacher."):
            continue
        if k.startswith("student."):
            new_key = k.replace("student.", "", 1)
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    print(f"Original parameter count: {len(state_dict)}")
    print(f"Cleaned parameter count: {len(new_state_dict)}")

    torch.save(new_state_dict, output_path)
    print(f"Saved cleaned checkpoint to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove teacher weights and strip the student prefix from a stage-2 checkpoint."
    )
    parser.add_argument(
        "input_path",
        help="Path to the original stage-2 checkpoint.",
    )
    parser.add_argument(
        "output_path",
        help="Path to save the cleaned test checkpoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    clean_checkpoint(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
