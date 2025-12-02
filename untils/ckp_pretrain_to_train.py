import torch
from collections import OrderedDict

# virpro_path = "/root/autodl-tmp/VirPro/ckp_gga/VirPro_pedestrian/VirPro_pedestrian_11.pkl"    # VirPro 预训练权重路径
# out_path    = "/root/autodl-tmp/GGA/ckp/pretrain/VirPro_pedestrian_11.pth"  # 只含 backbone.* 的新权重

virpro_path = "ckp/pretrain/VirPro_cyclist_19.pkl"    # VirPro 预训练权重路径
out_path    = "ckp/pretrain/VirPro_cyclist_19.pth"  # 只含 backbone.* 的新权重

def get_state_dict(ckpt):
    # 兼容 {'state_dict': ...} / {'model': ...} / 直接是 state_dict
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

# 1) 读取 VirPro
vir_raw = torch.load(virpro_path, map_location="cpu")
vir_sd  = strip_module(get_state_dict(vir_raw))

# 2) 选出 encoder.* 并改名前缀为 backbone.*
backbone_sd = OrderedDict()
for k, v in vir_sd.items():
    if k.startswith("encoder."):
        new_k = "backbone." + k[len("encoder."):]
        backbone_sd[new_k] = v

print(f"[INFO] collected {len(backbone_sd)} backbone tensors from VirPro.")

# 3) 保存为纯 state_dict（只含 backbone.*）
torch.save(backbone_sd, out_path)
print(f"[OK] saved backbone-only state_dict to: {out_path}")
