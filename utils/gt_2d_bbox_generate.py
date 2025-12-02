#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 KITTI label_2 中生成仅包含 Car 的 2D bbox 与置信度=1 的新标注文件。
- 仅处理 train set（由 train.txt 指定）的样本
- 每个输出文件与原文件同名（仍为 .txt），每行格式：
  Car <left> <top> <right> <bottom> 1
用法：
python kitti_car_2dbbox_only.py \
  --labels_dir /path/to/training/label_2 \
  --train_list /path/to/train.txt \
  --out_dir /path/to/output_dir
"""

import argparse
import os
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", required=True, help="KITTI training/label_2 目录")
    ap.add_argument("--train_list", required=True, help="train.txt（只含样本 id，每行如 000006）")
    ap.add_argument("--out_dir", required=True, help="输出目录，不存在会自动创建")
    ap.add_argument("--keep_empty", action="store_true",
                    help="若样本中没有 Car，是否仍生成空文件（默认跳过不生成）")
    return ap.parse_args()

def load_ids(train_list_path):
    ids = []
    with open(train_list_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # 兼容可能出现的诸如 '000006 ' 或 '\t000006'
            ids.append(s)
    return ids

def find_label_file(labels_dir, sample_id):
    """
    兼容两种命名：
    - 000006.txt（标准 KITTI）
    - 000006_.txt（你示例里也有这个变体）
    """
    p1 = Path(labels_dir) / f"{sample_id}.txt"
    if p1.exists():
        return p1
    p2 = Path(labels_dir) / f"{sample_id}_.txt"
    if p2.exists():
        return p2
    return None

def convert_one(label_path):
    """
    读取一个 KITTI label_2 文件，提取所有 Car 的 2D bbox，置信度写 1。
    KITTI 行格式：
    type, trunc, occ, alpha, left, top, right, bottom, ...
    我们保留：type(=Car), left, top, right, bottom, score(=1)
    """
    out_lines = []
    with open(label_path, "r", encoding="utf-8") as f:
        for raw in f:
            parts = raw.strip().split()
            if not parts:
                continue
            obj_type = parts[0]
            if obj_type != "Car":     # 只保留 Car；自动排除 Misc / DontCare 等
                continue
            if len(parts) < 8:
                # 防御式：格式异常就跳过
                continue
            try:
                left  = float(parts[4])
                top   = float(parts[5])
                right = float(parts[6])
                bottom= float(parts[7])
            except Exception:
                continue
            # 保留小数两位（可按需调整）
            out_lines.append(f"{left:.2f} {top:.2f} {right:.2f} {bottom:.2f}")
    return out_lines

def main():
    args = parse_args()
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = load_ids(args.train_list)
    missing, written, empty = 0, 0, 0

    for sid in ids:
        label_file = find_label_file(labels_dir, sid)
        if label_file is None:
            missing += 1
            continue
        lines = convert_one(label_file)
        if not lines and not args.keep_empty:
            empty += 1
            continue
        out_path = out_dir / f"{sid}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")
        written += 1

    print(f"完成：写出 {written} 个文件；缺失原标注 {missing} 个；因无 Car 而跳过 {empty} 个（use --keep_empty 可生成空文件）。")

if __name__ == "__main__":
    main()
