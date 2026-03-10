import os
import subprocess
import re
from collections import defaultdict
import yaml

# GT_DIR = "/mnt/md0/rajalakshmi/Unet_seg/real_data/catmaid_annotations/1738_GT/"
# PRED_DIR = "/mnt/md0/rajalakshmi/Unet_seg/finetuning_real_data/teasar_swc/1738/"
# GT_DIR = "/mnt/md0/rajalakshmi/Unet_seg/whole_pipeline_with_zarr_funlib/ground_truth_graph/"
# PRED_DIR = "/mnt/md0/rajalakshmi/Unet_seg/whole_pipeline_with_zarr_funlib/vesselformer/teasar_swc/"
# GT_DIR = "/mnt/md0/rajalakshmi/Unet_seg/real_data/catmaid_annotations/resampling_lisa/GT/"
# PRED_DIR = "//mnt/md0/rajalakshmi/Unet_seg/real_data/catmaid_annotations/resampling_lisa/output_pred/"


with open("eval_config.yaml", "r") as f:
    config = yaml.safe_load(f)

dataset_name = config["active_dataset"]

if dataset_name not in config["evaluation_by_dataset"]:
    raise ValueError(f"Dataset '{dataset_name}' not found in config.yaml")

GT_DIR = config["paths"]["gt_dir"]
PRED_DIR = config["paths"]["pred_dir"]
MAIN_SCRIPT = config["paths"]["main_script"]

eval_cfg = config["evaluation_by_dataset"][dataset_name]

MATCHING = eval_cfg["matching"]
MATCHING_DIST = eval_cfg["matching_dist"]
CANDIDATE_SELECTION = eval_cfg["candidate_selection"]
MAX_DISTANCE = eval_cfg["max_distance"]
RESAMPLE_TO_GT = eval_cfg["resample_to_gt"]

print(f"Using evaluation config for dataset: {dataset_name}")

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
        "--candidate_selection", CANDIDATE_SELECTION,
        "--max_distance", str(MAX_DISTANCE),
        "--smd"
    ]
    if RESAMPLE_TO_GT:
        cmd.append("--resample_to_gt")
    print(cmd)

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
