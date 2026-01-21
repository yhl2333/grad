import os
from pathlib import Path


# ====== 配置区 ======
LABEL_ROOT = "/home/yhl/ultralytics/datasets/VisDrone/labels_org"  # labels 根目录
SPLITS = ["train", "val", "test"]

# VisDrone 原始类 -> 新类
CLASS_MAP = {
    0: 0,  # pedestrian
    1: 0,  # people

    2: 1,  # bicycle
    9: 1,  # motor

    6: 2,  # tricycle
    7: 2,  # awning-tricycle

    3: 3,  # car
    4: 3,  # van
    5: 3,  # truck
    8: 3,  # bus
}

# 是否覆盖原文件（True）还是输出到 labels_converted
OVERWRITE = False
# ====================


def convert_one_file(src_path: Path, dst_path: Path):
    new_lines = []

    with open(src_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls = int(parts[0])
            if cls not in CLASS_MAP:
                continue

            new_cls = CLASS_MAP[cls]
            new_lines.append(
                f"{new_cls} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n"
            )

    if new_lines:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_path, "w") as f:
            f.writelines(new_lines)


def main():
    for split in SPLITS:
        src_dir = Path(LABEL_ROOT) / split
        

        if OVERWRITE:
            dst_dir = src_dir
        else:
            dst_dir = Path(LABEL_ROOT + "_converted") / split

        for label_file in src_dir.glob("*.txt"):
            dst_file = dst_dir / label_file.name
            convert_one_file(label_file, dst_file)



    print("✅ Label conversion finished.")


if __name__ == "__main__":
    main()
