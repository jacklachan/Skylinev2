import csv
from collections import defaultdict

LATENCY_CSV = "results/latency.csv"
DETECTIONS_CSV = "results/detections.csv"

STAGES = ["capture_ms", "yolov8_ms", "yoloworld_ms", "annotate_ms"]


def _stats(values):
    return min(values), sum(values) / len(values), max(values)


def main():
    stage_data = defaultdict(list)
    totals = []

    try:
        with open(LATENCY_CSV, newline="") as f:
            for row in csv.DictReader(f):
                total = 0.0
                for s in STAGES:
                    v = float(row[s])
                    stage_data[s].append(v)
                    total += v
                totals.append(total)
    except FileNotFoundError:
        print(f"No latency data yet: {LATENCY_CSV}")
    except Exception as e:
        print(f"Error reading {LATENCY_CSV}: {e}")

    if totals:
        print(f"=== Latency (ms) — {len(totals)} frames ===")
        print(f"{'stage':<20} {'min':>8} {'mean':>10} {'max':>8}")
        print("-" * 50)
        for s in STAGES:
            mn, mean, mx = _stats(stage_data[s])
            print(f"{s.replace('_ms', ''):<20} {mn:>8.1f} {mean:>10.1f} {mx:>8.1f}")
        mn, mean, mx = _stats(totals)
        print(f"{'total':<20} {mn:>8.1f} {mean:>10.1f} {mx:>8.1f}")

    class_conf = defaultdict(list)

    try:
        with open(DETECTIONS_CSV, newline="") as f:
            for row in csv.DictReader(f):
                class_conf[(row["model"], row["class"])].append(float(row["confidence"]))
    except FileNotFoundError:
        print(f"\nNo detection data yet: {DETECTIONS_CSV}")
    except Exception as e:
        print(f"Error reading {DETECTIONS_CSV}: {e}")

    if class_conf:
        total_detections = sum(len(v) for v in class_conf.values())
        print(f"\n=== Confidence per class — {total_detections} detections ===")
        print(f"{'model':<12} {'class':<22} {'min':>7} {'max':>7} {'count':>7}")
        print("-" * 60)
        for (model, cls), confs in sorted(class_conf.items()):
            print(f"{model:<12} {cls:<22} {min(confs):>7.3f} {max(confs):>7.3f} {len(confs):>7}")


if __name__ == "__main__":
    main()
