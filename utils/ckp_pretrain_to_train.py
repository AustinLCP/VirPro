import argparse
from collections import OrderedDict

import torch

def get_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "ema_state_dict", "module"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt

def strip_module(sd):
    new_sd = OrderedDict()
    for k, v in sd.items():
        new_sd[k[7:] if k.startswith("module.") else k] = v
    return new_sd

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a stage-1 VirPro checkpoint to a backbone-only training checkpoint."
    )
    parser.add_argument(
        "input_ckp_path",
        help="Path to the stage-1 pretrained checkpoint.",
    )
    parser.add_argument(
        "output_ckp_path",
        help="Path to save the converted backbone-only checkpoint.",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    vir_raw = torch.load(args.input_ckp_path, map_location="cpu")
    vir_sd = strip_module(get_state_dict(vir_raw))

    backbone_sd = OrderedDict()
    for k, v in vir_sd.items():
        if k.startswith("encoder."):
            new_k = "backbone." + k[len("encoder."):]
            backbone_sd[new_k] = v

    print(f"[INFO] collected {len(backbone_sd)} backbone tensors from VirPro.")

    torch.save(backbone_sd, args.output_ckp_path)
    print(f"[OK] saved backbone-only state_dict to: {args.output_ckp_path}")

if __name__ == "__main__":
    main()
