# import os
# from pathlib import Path

# import pandas as pd
# import matplotlib.pyplot as plt


# def load_results(file_path):
#     """
#     读取 Ultralytics 的 results.txt / results.csv 文件
#     自动兼容逗号分隔和空白分隔
#     """
#     file_path = Path(file_path)
#     if not file_path.exists():
#         raise FileNotFoundError(f"文件不存在: {file_path}")

#     # 先尝试按 csv 读取
#     try:
#         df = pd.read_csv(file_path)
#         if df.shape[1] > 1:
#             df.columns = [c.strip() for c in df.columns]
#             return df
#     except Exception:
#         pass

#     # 再尝试按空白分隔读取
#     try:
#         df = pd.read_csv(file_path, sep=r"\s+", engine="python")
#         df.columns = [c.strip() for c in df.columns]
#         return df
#     except Exception as e:
#         raise ValueError(f"无法解析文件: {file_path}\n错误信息: {e}")


# def find_column(df, candidates):
#     """
#     在 DataFrame 中查找匹配的列名
#     candidates: 候选列名列表
#     """
#     cols = list(df.columns)
#     for cand in candidates:
#         if cand in cols:
#             return cand

#     lower_map = {c.lower().strip(): c for c in cols}
#     for cand in candidates:
#         key = cand.lower().strip()
#         if key in lower_map:
#             return lower_map[key]

#     raise KeyError(f"未找到列名，候选为: {candidates}\n当前列名: {cols}")


# def plot_compare(before_file, after_file, save_dir):
#     """
#     before_file: 改进前 results.txt / results.csv 路径
#     after_file : 改进后 results.txt / results.csv 路径
#     save_dir   : 保存目录
#     """
#     save_dir = Path(save_dir)
#     save_dir.mkdir(parents=True, exist_ok=True)

#     df_before = load_results(before_file)
#     df_after = load_results(after_file)

#     # 只保留 6 张 loss 图：上面 3 张 train，下面 3 张 val
#     metric_map = [
#         ("train/box_loss", ["train/box_loss"]),
#         ("train/cls_loss", ["train/cls_loss"]),
#         ("train/dfl_loss", ["train/dfl_loss"]),
#         ("val/box_loss", ["val/box_loss"]),
#         ("val/cls_loss", ["val/cls_loss"]),
#         ("val/dfl_loss", ["val/dfl_loss"]),
#     ]

#     # 横轴优先用 epoch，没有就用行号
#     if "epoch" in df_before.columns:
#         x_before = df_before["epoch"]
#     else:
#         x_before = range(len(df_before))

#     if "epoch" in df_after.columns:
#         x_after = df_after["epoch"]
#     else:
#         x_after = range(len(df_after))

#     # 2 行 3 列
#     fig, axes = plt.subplots(2, 3, figsize=(12, 8))
#     axes = axes.flatten()

#     for ax, (title, candidates) in zip(axes, metric_map):
#         col_before = find_column(df_before, candidates)
#         col_after = find_column(df_after, candidates)

#         ax.plot(x_before, df_before[col_before], label="Baseline Model", linewidth=2)
#         ax.plot(x_after, df_after[col_after], label="Improved Model", linewidth=2)

#         ax.set_title(title, fontsize=12)
#         ax.grid(True, linestyle="--", alpha=0.3)
#         ax.tick_params(labelsize=10)

#     # 总图例
#     handles, labels = axes[0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=12)

#     plt.tight_layout(rect=[0, 0, 1, 0.94])

#     save_path = save_dir / "loss_compare.png"
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close()

#     print(f"对比图已保存到: {save_path}")


# if __name__ == "__main__":
#     before_file = "/disk2/yhl/ultralytics/runs/detect/v6.4_pl3/results.csv"
#     after_file = "/disk2/yhl/ultralytics/runs/detect/v6.4_n_p2_shanp5_cl3_SimAM_EUCB/results.csv"
#     save_dir = "/disk2/yhl/ultralytics/experient_fig/loss"

#     plot_compare(before_file, after_file, save_dir)















# import os
# from pathlib import Path

# import pandas as pd
# import matplotlib.pyplot as plt


# def load_results(file_path):
#     file_path = Path(file_path)
#     if not file_path.exists():
#         raise FileNotFoundError(f"文件不存在: {file_path}")

#     try:
#         df = pd.read_csv(file_path)
#         if df.shape[1] > 1:
#             df.columns = [c.strip() for c in df.columns]
#             return df
#     except Exception:
#         pass

#     try:
#         df = pd.read_csv(file_path, sep=r"\s+", engine="python")
#         df.columns = [c.strip() for c in df.columns]
#         return df
#     except Exception as e:
#         raise ValueError(f"无法解析文件: {file_path}\n错误信息: {e}")


# def find_column(df, candidates):
#     cols = list(df.columns)
#     for cand in candidates:
#         if cand in cols:
#             return cand

#     lower_map = {c.lower().strip(): c for c in cols}
#     for cand in candidates:
#         key = cand.lower().strip()
#         if key in lower_map:
#             return lower_map[key]

#     raise KeyError(f"未找到列名，候选为: {candidates}\n当前列名: {cols}")


# def plot_pr_compare(before_file, after_file, save_dir):
#     save_dir = Path(save_dir)
#     save_dir.mkdir(parents=True, exist_ok=True)

#     df_before = load_results(before_file)
#     df_after = load_results(after_file)

#     metric_map = [
#         ("metrics/precision(B)", ["metrics/precision(B)", "metrics/precision"]),
#         ("metrics/recall(B)", ["metrics/recall(B)", "metrics/recall"]),
#     ]

#     x_before = df_before["epoch"] if "epoch" in df_before.columns else range(len(df_before))
#     x_after = df_after["epoch"] if "epoch" in df_after.columns else range(len(df_after))

#     fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
#     axes = axes.flatten()

#     for ax, (title, candidates) in zip(axes, metric_map):
#         col_before = find_column(df_before, candidates)
#         col_after = find_column(df_after, candidates)

#         ax.plot(x_before, df_before[col_before], label="Baseline Model", linewidth=2)
#         ax.plot(x_after, df_after[col_after], label="Improved Model", linewidth=2)

#         ax.set_title(title, fontsize=12)
#         ax.grid(True, linestyle="--", alpha=0.3)
#         ax.tick_params(labelsize=10)

#     handles, labels = axes[0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=11)

#     plt.tight_layout(rect=[0, 0, 1, 0.90])

#     save_path = save_dir / "PR_compare.png"
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close()

#     print(f"对比图已保存到: {save_path}")


# if __name__ == "__main__":
#     before_file = "/disk2/yhl/ultralytics/runs/detect/v6.4_pl3/results.csv"
#     after_file = "/disk2/yhl/ultralytics/runs/detect/v6.4_n_p2_shanp5_cl3_SimAM_EUCB/results.csv"
#     save_dir = "/disk2/yhl/ultralytics/experient_fig/loss"

#     plot_pr_compare(before_file, after_file, save_dir)












import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_results(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    try:
        df = pd.read_csv(file_path)
        if df.shape[1] > 1:
            df.columns = [c.strip() for c in df.columns]
            return df
    except Exception:
        pass

    try:
        df = pd.read_csv(file_path, sep=r"\s+", engine="python")
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        raise ValueError(f"无法解析文件: {file_path}\n错误信息: {e}")


def find_column(df, candidates):
    cols = list(df.columns)
    for cand in candidates:
        if cand in cols:
            return cand

    lower_map = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        key = cand.lower().strip()
        if key in lower_map:
            return lower_map[key]

    raise KeyError(f"未找到列名，候选为: {candidates}\n当前列名: {cols}")


def plot_map_compare(before_file, after_file, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df_before = load_results(before_file)
    df_after = load_results(after_file)

    metric_map = [
        ("metrics/mAP50(B)", ["metrics/mAP50(B)", "metrics/mAP_0.5", "metrics/mAP50"]),
        ("metrics/mAP50-95(B)", ["metrics/mAP50-95(B)", "metrics/mAP_0.5:0.95", "metrics/mAP50-95"]),
    ]

    x_before = df_before["epoch"] if "epoch" in df_before.columns else range(len(df_before))
    x_after = df_after["epoch"] if "epoch" in df_after.columns else range(len(df_after))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes = axes.flatten()

    for ax, (title, candidates) in zip(axes, metric_map):
        col_before = find_column(df_before, candidates)
        col_after = find_column(df_after, candidates)

        ax.plot(x_before, df_before[col_before], label="Baseline Model", linewidth=2)
        ax.plot(x_after, df_after[col_after], label="Improved Model", linewidth=2)

        ax.set_title(title, fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(labelsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    save_path = save_dir / "mAP_compare.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"对比图已保存到: {save_path}")


if __name__ == "__main__":
    before_file = "/disk2/yhl/ultralytics/runs/detect/v6.4_pl3/results.csv"
    after_file = "/disk2/yhl/ultralytics/runs/detect/v6.4_n_p2_shanp5_cl3_SimAM_EUCB/results.csv"
    save_dir = "/disk2/yhl/ultralytics/experient_fig/loss"

    plot_map_compare(before_file, after_file, save_dir)