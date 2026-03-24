import csv
from pathlib import Path
from collections import defaultdict

# =========================
# 路径配置
# =========================
PRED_PATH = Path("results/exp14_test/cross_match.csv")
GT_PATH = Path("results/exp13_reidkalam/cross_match_gt.csv")

REPORT_TXT = Path("results/exp14_test/eval_report.txt")
PER_FRAME_CSV = Path("results/exp14_test/eval_per_frame.csv")

# 300帧之前：默认 frame_id < 300
FRAME_CUTOFF = 300
INCLUDE_CUTOFF = False   # True 表示 <=300；False 表示 <300


# =========================
# 读取三元组
# 支持：
# 1) 带表头: frame_id,track_id_a,track_id_b
# 2) 无表头: 586,151,112
# =========================
def load_triplets(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    triplets = []

    with open(path, "r", encoding="utf-8-sig") as f:
        first_line = f.readline().strip()
        if not first_line:
            return triplets

        has_header = False
        lower = first_line.lower()
        if "frame" in lower or "track" in lower:
            has_header = True

    with open(path, "r", encoding="utf-8-sig") as f:
        if has_header:
            reader = csv.DictReader(f)
            # 尝试兼容不同列名
            fieldnames = [x.strip() for x in (reader.fieldnames or [])]

            def pick_key(candidates):
                for c in candidates:
                    if c in fieldnames:
                        return c
                return None

            kf = pick_key(["frame_id", "frame", "fid"])
            ka = pick_key(["track_id_a", "a_id", "track_a"])
            kb = pick_key(["track_id_b", "b_id", "track_b"])

            if kf is None or ka is None or kb is None:
                raise ValueError(
                    f"{path} 表头无法识别，需要包含 frame_id/track_id_a/track_id_b 或同义字段。"
                )

            for row in reader:
                try:
                    frame_id = int(float(row[kf]))
                    track_id_a = int(float(row[ka]))
                    track_id_b = int(float(row[kb]))
                    triplets.append((frame_id, track_id_a, track_id_b))
                except Exception:
                    continue
        else:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = [x.strip() for x in line.split(",")]
                if len(parts) < 3:
                    continue
                try:
                    frame_id = int(float(parts[0]))
                    track_id_a = int(float(parts[1]))
                    track_id_b = int(float(parts[2]))
                    triplets.append((frame_id, track_id_a, track_id_b))
                except Exception:
                    continue

    # 去重
    triplets = sorted(set(triplets))
    return triplets


# =========================
# 按帧组织
# =========================
def group_by_frame(triplets):
    out = defaultdict(set)
    for frame_id, a, b in triplets:
        out[frame_id].add((a, b))
    return dict(out)


# =========================
# 计算一组评估指标
# 只在指定 frames 上评估
# =========================
def evaluate_subset(pred_triplets, gt_triplets, frames_to_eval):
    pred_set = set([x for x in pred_triplets if x[0] in frames_to_eval])
    gt_set = set([x for x in gt_triplets if x[0] in frames_to_eval])

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # 逐帧完全一致率
    pred_by_frame = group_by_frame(pred_set)
    gt_by_frame = group_by_frame(gt_set)

    exact_match_frames = 0
    per_frame_rows = []

    for frame_id in sorted(frames_to_eval):
        pred_pairs = pred_by_frame.get(frame_id, set())
        gt_pairs = gt_by_frame.get(frame_id, set())

        tp_f = len(pred_pairs & gt_pairs)
        fp_f = len(pred_pairs - gt_pairs)
        fn_f = len(gt_pairs - pred_pairs)

        p_f = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else 0.0
        r_f = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0.0
        f1_f = (2 * p_f * r_f / (p_f + r_f)) if (p_f + r_f) > 0 else 0.0

        exact = int(pred_pairs == gt_pairs)
        exact_match_frames += exact

        per_frame_rows.append({
            "frame_id": frame_id,
            "gt_count": len(gt_pairs),
            "pred_count": len(pred_pairs),
            "tp": tp_f,
            "fp": fp_f,
            "fn": fn_f,
            "precision": p_f,
            "recall": r_f,
            "f1": f1_f,
            "exact_match": exact,
        })

    frame_exact_match_rate = exact_match_frames / len(frames_to_eval) if frames_to_eval else 0.0

    metrics = {
        "num_frames": len(frames_to_eval),
        "num_gt_pairs": len(gt_set),
        "num_pred_pairs": len(pred_set),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "frame_exact_match_rate": frame_exact_match_rate,
    }

    return metrics, per_frame_rows


# =========================
# 打印格式化
# =========================
def format_metrics(title, m):
    lines = []
    lines.append(f"[{title}]")
    lines.append(f"  标注帧数             : {m['num_frames']}")
    lines.append(f"  GT匹配条数           : {m['num_gt_pairs']}")
    lines.append(f"  预测匹配条数         : {m['num_pred_pairs']}")
    lines.append(f"  TP / FP / FN         : {m['tp']} / {m['fp']} / {m['fn']}")
    lines.append(f"  Precision            : {m['precision']:.6f}")
    lines.append(f"  Recall               : {m['recall']:.6f}")
    lines.append(f"  F1                   : {m['f1']:.6f}")
    lines.append(f"  Frame Exact Match    : {m['frame_exact_match_rate']:.6f}")
    return "\n".join(lines)


# =========================
# 主程序
# =========================
def main():
    pred_triplets = load_triplets(PRED_PATH)
    gt_triplets = load_triplets(GT_PATH)

    print(f"[Info] 预测条数: {len(pred_triplets)}")
    print(f"[Info] GT条数  : {len(gt_triplets)}")

    gt_frames = sorted(set(f for f, _, _ in gt_triplets))

    if INCLUDE_CUTOFF:
        frames_before_300 = sorted([f for f in gt_frames if f <= FRAME_CUTOFF])
        range_name = f"已标注帧中 frame <= {FRAME_CUTOFF}"
    else:
        frames_before_300 = sorted([f for f in gt_frames if f < FRAME_CUTOFF])
        range_name = f"已标注帧中 frame < {FRAME_CUTOFF}"

    frames_all_labeled = gt_frames

    metrics_before_300, per_frame_before_300 = evaluate_subset(
        pred_triplets, gt_triplets, frames_before_300
    )
    metrics_all, per_frame_all = evaluate_subset(
        pred_triplets, gt_triplets, frames_all_labeled
    )

    report = []
    report.append("Cross-view Match Evaluation")
    report.append(f"PRED_PATH = {PRED_PATH}")
    report.append(f"GT_PATH   = {GT_PATH}")
    report.append("")
    report.append("说明：只在 GT 已标注的帧上评估。")
    report.append("")
    report.append(format_metrics(range_name, metrics_before_300))
    report.append("")
    report.append(format_metrics("所有已标注帧", metrics_all))
    report_text = "\n".join(report)

    print("\n" + report_text)

    REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[Info] 报告已保存到: {REPORT_TXT}")

    # 导出逐帧明细（两段合并到一个文件里）
    with open(PER_FRAME_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "subset", "frame_id", "gt_count", "pred_count",
            "tp", "fp", "fn", "precision", "recall", "f1", "exact_match"
        ])

        subset_name = "before_300" if not INCLUDE_CUTOFF else "le_300"
        for row in per_frame_before_300:
            writer.writerow([
                subset_name,
                row["frame_id"],
                row["gt_count"],
                row["pred_count"],
                row["tp"],
                row["fp"],
                row["fn"],
                f"{row['precision']:.6f}",
                f"{row['recall']:.6f}",
                f"{row['f1']:.6f}",
                row["exact_match"],
            ])

        for row in per_frame_all:
            writer.writerow([
                "all_labeled",
                row["frame_id"],
                row["gt_count"],
                row["pred_count"],
                row["tp"],
                row["fp"],
                row["fn"],
                f"{row['precision']:.6f}",
                f"{row['recall']:.6f}",
                f"{row['f1']:.6f}",
                row["exact_match"],
            ])

    print(f"[Info] 逐帧明细已保存到: {PER_FRAME_CSV}")


if __name__ == "__main__":
    main()