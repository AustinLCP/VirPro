import torch
import argparse

###############################################################################
# 训练生成的权重文件包含了 teacher 和 student 的所有权重，现在我们只保留 student 的 权重 #
###############################################################################

def clean_checkpoint(input_path, output_path):
    # 加载原始权重
    ckpt = torch.load(input_path, map_location='cpu')

    # 有些文件外层是 {'state_dict': ...}
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    new_state_dict = {}
    for k, v in state_dict.items():
        # 删除所有以 "teacher." 开头的键
        if k.startswith('teacher.'):
            continue
        # 如果是 student 的 key，去掉前缀 "student."
        elif k.startswith('student.'):
            new_key = k.replace('student.', '', 1)
            new_state_dict[new_key] = v
        else:
            # 其他 key（例如 backbone/neck/head 本身）原样保留
            new_state_dict[k] = v

    print(f"原始参数数量: {len(state_dict)}")
    print(f"清理后参数数量: {len(new_state_dict)}")

    # 保存清理后的权重
    torch.save(new_state_dict, output_path)
    print(f"已保存到: {output_path}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Remove teacher keys and strip student. prefix from checkpoint")
    # parser.add_argument("--input", required=True, help="原始权重文件路径 (.pth)")
    # parser.add_argument("--output", required=True, help="输出的清理后权重路径 (.pth)")
    # args = parser.parse_args()

    input_path = "ckp/VirPro_GGA_PDG_2d_gt_bbox_finetune_35.pth"
    output_path = "test_ckp/test_ckp.pth"

    clean_checkpoint(input_path, output_path)
