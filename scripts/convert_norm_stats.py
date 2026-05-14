#!/usr/bin/env python3
"""Convert LeRobot dataset stats.json to π₀ norm_stats.json with OBS_INDICES reordering.

This script reads the dataset's meta/stats.json (which has stats in the original 30D
interleaved state order) and generates the norm_stats.json that π₀ expects (28D
reordered state order).

The 30D raw state order (1-based indexing from info.json):
  1-14:  [left_joint_1, right_joint_1, left_joint_2, right_joint_2, ..., left_joint_7, right_joint_7]
  15-26: [left_ee_x, right_ee_x, left_ee_y, right_ee_y, ..., left_ee_rz, right_ee_rz]
  27-28: [left_gripper_state, right_gripper_state]
  29-30: [left_gripper_cmd, right_gripper_cmd]

The 28D reordered state (after OBS_INDICES filtering and reordering):
  [left_joint(7), right_joint(7), left_ee(6), right_ee(6), left_gripper_state(1), right_gripper_state(1)]

OBS_INDICES (1-based): 1,3,5,7,9,11,13,2,4,6,8,10,12,14,15,17,19,21,23,25,16,18,20,22,24,26,27,28

Usage:
    python scripts/convert_norm_stats.py \
        --dataset-path /path/to/dataset \
        --output-path /path/to/norm_stats.json
"""

import argparse
import json
import numpy as np
from pathlib import Path


# OBS_INDICES: 1-based indices into the original 30D state vector.
# These select and reorder the 30D state into the 28D grouped order.
# - 1,3,5,7,9,11,13 → left_joint_1..7 (odd indices from interleaved joints)
# - 2,4,6,8,10,12,14 → right_joint_1..7 (even indices from interleaved joints)
# - 15,17,19,21,23,25 → left_ee_x,y,z,rx,ry,rz (odd indices from interleaved EE)
# - 16,18,20,22,24,26 → right_ee_x,y,z,rx,ry,rz (even indices from interleaved EE)
# - 27 → left_gripper_state
# - 28 → right_gripper_state
# (indices 29,30 = left/right_gripper_cmd are dropped)
OBS_INDICES = [1, 3, 5, 7, 9, 11, 13, 2, 4, 6, 8, 10, 12, 14, 15, 17, 19, 21, 23, 25, 16, 18, 20, 22, 24, 26, 27, 28]


def reorder_stats(values: list, obs_indices: list[int]) -> list:
    """Reorder a list of values according to OBS_INDICES (1-based).
    
    Args:
        values: List of values in the original 30D order.
        obs_indices: 1-based indices to select and reorder.
    
    Returns:
        Reordered list of values.
    """
    # Convert 1-based to 0-based
    zero_based = [i - 1 for i in obs_indices]
    return [values[i] for i in zero_based]


def convert_norm_stats(dataset_path: str, output_path: str | None = None):
    """Convert dataset stats.json to π₀ norm_stats.json.
    
    Args:
        dataset_path: Path to the dataset directory (containing meta/stats.json).
        output_path: Path to write the output norm_stats.json. If None, writes to
                     dataset_path/norm_stats.json.
    """
    dataset_path = Path(dataset_path)
    stats_path = dataset_path / "meta" / "stats.json"
    
    if not stats_path.exists():
        raise FileNotFoundError(f"Dataset stats not found at {stats_path}")
    
    with open(stats_path, "r") as f:
        dataset_stats = json.load(f)
    
    # Extract state stats from the 30D original order
    state_stats = dataset_stats["observation.state"]
    action_stats = dataset_stats["action"]
    
    # Reorder state stats from 30D interleaved to 28D grouped
    norm_stats = {"norm_stats": {"state": {}, "actions": {}}}
    
    # State stats to reorder
    for stat_name in ["mean", "std", "q01", "q99"]:
        if stat_name in state_stats:
            original_values = state_stats[stat_name]
            reordered_values = reorder_stats(original_values, OBS_INDICES)
            norm_stats["norm_stats"]["state"][stat_name] = reordered_values
    
    # Action stats: 14D, no reordering needed (already in correct order)
    for stat_name in ["mean", "std", "q01", "q99"]:
        if stat_name in action_stats:
            norm_stats["norm_stats"]["actions"][stat_name] = action_stats[stat_name]
    
    # Determine output path
    if output_path is None:
        output_path = str(dataset_path / "norm_stats.json")
    
    with open(output_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    
    print(f"✅ Converted norm_stats.json written to: {output_path}")
    
    # Print verification
    print(f"\n📊 State stats verification (28D reordered):")
    print(f"   mean: {norm_stats['norm_stats']['state']['mean']}")
    print(f"   std:  {norm_stats['norm_stats']['state']['std']}")
    
    print(f"\n📊 Action stats (14D, no reorder):")
    print(f"   mean: {norm_stats['norm_stats']['actions']['mean']}")
    print(f"   std:  {norm_stats['norm_stats']['actions']['std']}")
    
    # Verify the reordering by checking known patterns
    _verify_reordering(state_stats, norm_stats["norm_stats"]["state"])


def _verify_reordering(original_state: dict, reordered_state: dict):
    """Verify the reordering is correct by checking known value patterns."""
    orig_mean = original_state["mean"]
    reord_mean = reordered_state["mean"]
    
    print(f"\n🔍 Reordering verification:")
    
    # Check left_joint: should be indices 0,2,4,6,8,10,12 from original (0-based)
    expected_left_joint = [orig_mean[0], orig_mean[2], orig_mean[4], orig_mean[6], 
                           orig_mean[8], orig_mean[10], orig_mean[12]]
    actual_left_joint = reord_mean[:7]
    
    left_joint_ok = all(abs(e - a) < 1e-4 for e, a in zip(expected_left_joint, actual_left_joint))
    print(f"   left_joint(7):  {'✅' if left_joint_ok else '❌'}")
    if not left_joint_ok:
        print(f"     Expected: {[round(v, 6) for v in expected_left_joint]}")
        print(f"     Actual:   {[round(v, 6) for v in actual_left_joint]}")
    
    # Check right_joint: should be indices 1,3,5,7,9,11,13 from original (0-based)
    expected_right_joint = [orig_mean[1], orig_mean[3], orig_mean[5], orig_mean[7],
                            orig_mean[9], orig_mean[11], orig_mean[13]]
    actual_right_joint = reord_mean[7:14]
    
    right_joint_ok = all(abs(e - a) < 1e-4 for e, a in zip(expected_right_joint, actual_right_joint))
    print(f"   right_joint(7): {'✅' if right_joint_ok else '❌'}")
    if not right_joint_ok:
        print(f"     Expected: {[round(v, 6) for v in expected_right_joint]}")
        print(f"     Actual:   {[round(v, 6) for v in actual_right_joint]}")
    
    # Check left_ee: should be indices 14,16,18,20,22,24 from original (0-based)
    expected_left_ee = [orig_mean[14], orig_mean[16], orig_mean[18], orig_mean[20],
                       orig_mean[22], orig_mean[24]]
    actual_left_ee = reord_mean[14:20]
    
    left_ee_ok = all(abs(e - a) < 1e-4 for e, a in zip(expected_left_ee, actual_left_ee))
    print(f"   left_ee(6):     {'✅' if left_ee_ok else '❌'}")
    if not left_ee_ok:
        print(f"     Expected: {[round(v, 6) for v in expected_left_ee]}")
        print(f"     Actual:   {[round(v, 6) for v in actual_left_ee]}")
    
    # Check right_ee: should be indices 15,17,19,21,23,25 from original (0-based)
    expected_right_ee = [orig_mean[15], orig_mean[17], orig_mean[19], orig_mean[21],
                        orig_mean[23], orig_mean[25]]
    actual_right_ee = reord_mean[20:26]
    
    right_ee_ok = all(abs(e - a) < 1e-4 for e, a in zip(expected_right_ee, actual_right_ee))
    print(f"   right_ee(6):    {'✅' if right_ee_ok else '❌'}")
    if not right_ee_ok:
        print(f"     Expected: {[round(v, 6) for v in expected_right_ee]}")
        print(f"     Actual:   {[round(v, 6) for v in actual_right_ee]}")
    
    # Check grippers: should be indices 26,27 from original (0-based)
    expected_grippers = [orig_mean[26], orig_mean[27]]
    actual_grippers = reord_mean[26:28]
    
    grippers_ok = all(abs(e - a) < 1e-4 for e, a in zip(expected_grippers, actual_grippers))
    print(f"   grippers(2):    {'✅' if grippers_ok else '❌'}")
    if not grippers_ok:
        print(f"     Expected: {[round(v, 6) for v in expected_grippers]}")
        print(f"     Actual:   {[round(v, 6) for v in actual_grippers]}")
    
    all_ok = left_joint_ok and right_joint_ok and left_ee_ok and right_ee_ok and grippers_ok
    print(f"\n   Overall: {'✅ All checks passed!' if all_ok else '❌ Some checks failed!'}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot dataset stats.json to π₀ norm_stats.json with OBS_INDICES reordering"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset directory (containing meta/stats.json)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to write the output norm_stats.json (default: <dataset-path>/norm_stats.json)",
    )
    args = parser.parse_args()
    
    convert_norm_stats(args.dataset_path, args.output_path)


if __name__ == "__main__":
    main()
