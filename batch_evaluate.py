import os
import subprocess
import re
from collections import defaultdict


GT_DIR = "/mnt/md0/rajalakshmi/Unet_seg/whole_pipeline_with_zarr_funlib/ground_truth_graph/"
PRED_DIR = "/mnt/md0/rajalakshmi/Unet_seg/experiments/random_crops/teasar_swc/"
MAIN_SCRIPT = "main.py"

MATCHING = "hierarchical"
MATCHING_DIST = "fixed"


pattern = re.compile(r"^(\w+):\s+([-+]?[0-9]*\.?[0-9]+)$")


results_accumulator = defaultdict(list)

for filename in os.listdir(GT_DIR):
    if not filename.endswith(".swc"):
        continue

    gt_path = os.path.join(GT_DIR, filename)
    pred_path = os.path.join(PRED_DIR, filename)

    if not os.path.exists(pred_path):
        print(f"Prediction missing for {filename}, skipping.")
        continue


    cmd = [
        "python", MAIN_SCRIPT,
        "--gt_fn", gt_path,
        "--pred_fn", pred_path,
        "--matching", MATCHING,
        "--matching_dist", MATCHING_DIST,
        "--smd"
    ]

    try:

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            match = pattern.match(line)
            if match:
                key, value = match.groups()
                results_accumulator[key].append(float(value))
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating {filename}: {e.stderr}")


print("\nMean Evaluation Results")
for key, values in results_accumulator.items():
    mean_val = sum(values) / len(values)
    print(f"{key}: {mean_val}")
