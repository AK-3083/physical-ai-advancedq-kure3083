"""正解との一対一対応で探索結果を測定し、独立した色マーカー15試行を実行する。"""

import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import numpy as np

from scripts.mountain_scene import generate_mountain
from sos.reporting import write_report


def point_distance(first, second):
    """同じ次元の有限なXYまたはXYZ地点間の距離をメートルで返す。"""
    if ("z" in first) != ("z" in second):
        raise ValueError("Evaluation coordinate dimensions do not match")
    axes = ("x", "y", "z") if "z" in first else ("x", "y")
    values = np.array([[first[axis], second[axis]] for axis in axes], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Evaluation coordinates must be finite")
    return float(np.linalg.norm(values[:, 0] - values[:, 1]))


def matching(truth, estimates, radius=1.0):
    """半径内の一対一対応を増加路で最大化する。総誤差最小化は保証しない。"""
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Association radius must be positive")
    edges = []
    for marker in truth:
        distances = [point_distance(marker, item) for item in estimates]
        edges.append(sorted((distance, index) for index, distance in enumerate(distances)
                            if distance <= radius))
    owners = {}

    def assign(marker_index, visited):
        for _, estimate_index in edges[marker_index]:
            if estimate_index in visited:
                continue
            visited.add(estimate_index)
            if estimate_index not in owners or assign(owners[estimate_index], visited):
                owners[estimate_index] = marker_index
                return True
        return False

    for marker_index in range(len(truth)):
        assign(marker_index, set())
    return [{"marker_index": marker_index, "estimate_index": estimate_index,
             "error_m": float(np.hypot(truth[marker_index]["x"] - estimates[estimate_index]["x"],
                                      truth[marker_index]["y"] - estimates[estimate_index]["y"]))}
            for estimate_index, marker_index in sorted(owners.items())]


def evaluate(truth, report):
    """1m以内の対応で発見・誤報・誤差・重複・誤統合を集計し、状態も保持する。"""
    markers = truth["markers"]
    matches = matching(markers, report["sites"])
    labels = []
    for detection in report["detections"]:
        distances = [point_distance(detection, item) for item in markers]
        labels.append(int(np.argmin(distances)) if distances and min(distances) <= 1.0 else None)
    raw_counts = [labels.count(index) for index in range(len(markers))]
    site_counts = [sum(labels[index] == marker for index in site["detection_indices"])
                   for site in report["sites"] for marker in range(len(markers))]
    false_merges = sum(
        len({labels[index] for index in site["detection_indices"] if labels[index] is not None}) > 1
        for site in report["sites"]
    )
    errors = [item["error_m"] for item in matches]
    altitude_errors = [
        abs(markers[item["marker_index"]]["z"] - report["sites"][item["estimate_index"]]["z"])
        for item in matches if "z" in markers[item["marker_index"]]
    ]
    return {
        "seed": truth["seed"], "close_pair": truth.get("close_pair", False),
        "status": report["status"], "marker_count": len(markers),
        "matched_count": len(matches), "missed_count": len(markers) - len(matches),
        "false_reports": len(report["sites"]) - len(matches),
        "raw_detection_count": len(report["detections"]), "site_count": len(report["sites"]),
        "raw_duplicate_count": sum(max(0, count - 1) for count in raw_counts),
        "consolidated_duplicate_count": sum(
            max(0, sum(count > 0 for count in site_counts[index::len(markers)]) - 1)
            for index in range(len(markers))
        ) if markers else 0,
        "false_merge_count": int(false_merges), "matched_errors_m": errors,
        "matches": matches,
        **({"matched_altitude_errors_m": altitude_errors} if altitude_errors else {}),
    }


def stop_simulator(process):
    """専用セッションで起動したシミュレータをINT→TERM→KILLの順に停止する。"""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            logging.warning("Simulator did not stop on SIGINT; terminating owned process group")
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)


def color_comparison(truth, output):
    """実撮影の同じ色候補について、色のみとCNN採用を正解座標で比較する。"""
    scans_path = output / "scans.jsonl"
    scans = ([json.loads(line) for line in scans_path.read_text().splitlines()]
             if scans_path.exists() else [])
    search = [json.loads(line) for line in (output / "search-frames.jsonl").read_text().splitlines()]
    threshold = json.loads(Path("models/color/metrics.json").read_text())["threshold"]
    result = {key: 0 for key in ("color_only_false_candidates", "cnn_false_candidates",
                                "visible_marker_candidates", "cnn_marker_candidates",
                                "same_color_only_reports")}
    by_kind = {}
    for scan in scans:
        for observation in scan["observations"]:
            if (not any(point_distance(observation, marker) <= 1 for marker in truth["markers"])
                    and any(point_distance(observation, item) <= 1 for item in truth["negatives"])):
                result["same_color_only_reports"] += 1
    for scan in search:
        for candidate in scan["candidates"]:
            positive = any(point_distance(candidate, marker) <= 1 for marker in truth["markers"])
            accepted = candidate["score"] >= threshold
            result["visible_marker_candidates" if positive else "color_only_false_candidates"] += 1
            if accepted:
                result["cnn_marker_candidates" if positive else "cnn_false_candidates"] += 1
            if not positive:
                nearby = [item for item in truth["negatives"] if point_distance(candidate, item) <= 1]
                kind = nearby[0]["kind"] if nearby else "background"
                category = by_kind.setdefault(kind, {"color_only": 0, "cnn": 0})
                category["color_only"] += 1
                category["cnn"] += int(accepted)
    result["negative_categories"] = by_kind
    return result


def run_suite(directory, layout_variations=False):
    """標準15試行または変更5配置を新規フォルダで評価し、目標未達なら失敗する。"""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    cases = ([(501+index, 2, index+1) for index in range(5)] if layout_variations
             else [(seed, 2 if seed < 411 else 0, 0) for seed in range(401, 416)])
    results = []
    for seed, count, layout in cases:
        scene = directory / "scene"
        generate_mountain(scene, seed + 100, count, layout)
        output = directory / f"seed-{seed}"
        environment = dict(os.environ, SOS_WORLD=str(scene / "search.world"))
        started = time.monotonic()
        with (directory / f"seed-{seed}.sim.log").open("w") as log:
            simulator = subprocess.Popen(
                ["ros2", "launch", "/workspace/sos/scripts/sim.launch.py"],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=environment,
            )
            try:
                with (directory / f"seed-{seed}.mission.log").open("w") as mission_log:
                    command = ["/opt/sos/train/bin/python", "-m", "sos.mission", "--output", str(output),
                               "--terrain", str(scene / "terrain.json")]
                    mission = subprocess.run(
                        command, stdout=mission_log, stderr=subprocess.STDOUT,
                        timeout=12000,
                    )
                if mission.returncode != 0:
                    logging.error("Mission seed=%s failed with exit code %s", seed, mission.returncode)
            except subprocess.TimeoutExpired:
                logging.error("Mission seed=%s exceeded process timeout", seed)
                write_report(output, [], "failed", "Mission process exceeded evaluation timeout")
            finally:
                stop_simulator(simulator)
        if not (output / "report.json").exists():
            logging.error("Mission seed=%s produced no report", seed)
            write_report(output, [], "failed", "Mission process produced no report; inspect mission log")
        shutil.copyfile(scene / "truth.json", output / "truth.json")
        truth = json.loads((output / "truth.json").read_text())
        report = json.loads((output / "report.json").read_text())
        result = evaluate(truth, report)
        result["layout"] = layout
        if report["status"] == "complete":
            result["color_comparison"] = color_comparison(truth, output)
        result["wall_seconds"] = time.monotonic() - started
        if (output / "flight-metrics.json").exists():
            result["flight"] = json.loads((output / "flight-metrics.json").read_text())
        results.append(result)
        (output / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (directory / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
    totals = {
        key: sum(result[key] for result in results)
        for key in ["marker_count", "matched_count", "missed_count", "false_reports",
                    "raw_duplicate_count", "consolidated_duplicate_count", "false_merge_count"]
    }
    errors = [error for result in results for error in result["matched_errors_m"]]
    totals.update({
        "mode": "mountain", "scheme": "color", "layout_variations": layout_variations,
        "run_count": len(results), "completed_count": sum(item["status"] == "complete" for item in results),
        "recall": totals["matched_count"] / totals["marker_count"],
        "matched_within_0_5m_fraction": sum(error <= 0.5 for error in errors) / len(errors) if errors else None,
        "matched_within_1m_fraction": sum(error <= 1 for error in errors) / len(errors) if errors else None,
        "maximum_matched_error_m": max(errors) if errors else None,
        "empty_scene_false_reports": sum(item["false_reports"] for item in results if item["marker_count"] == 0),
        "total_wall_seconds": sum(item["wall_seconds"] for item in results),
    })
    totals["targets_met"] = (
        totals["recall"] >= 0.9 and totals["empty_scene_false_reports"] <= 1
        and bool(errors)
        and sum(error <= 1 for error in errors) / len(errors) >= 0.9
        and totals["completed_count"] == len(results) and totals["false_merge_count"] == 0
        and totals["consolidated_duplicate_count"] < totals["raw_duplicate_count"]
    )
    totals["minimum_flight_clearance_m"] = min(
        (item["flight"]["minimum_airborne_clearance_m"] for item in results if "flight" in item),
        default=None,
    )
    totals["summit_and_return_count"] = sum(
        item.get("flight", {}).get("maximum_altitude_m", 0) >= 20
        and item["status"] == "complete" for item in results
    )
    totals["targets_met"] = totals["targets_met"] and totals["summit_and_return_count"] == len(results)
    totals["return_method"] = "direct"
    totals["direct_return_count"] = sum(
        item.get("flight", {}).get("return_method") == "direct" for item in results)
    comparisons = [item["color_comparison"] for item in results if "color_comparison" in item]
    totals["color_comparison"] = {
        key: sum(item[key] for item in comparisons)
        for key in ("color_only_false_candidates", "cnn_false_candidates",
                    "visible_marker_candidates", "cnn_marker_candidates", "same_color_only_reports")
    }
    categories = {}
    for comparison in comparisons:
        for kind, values in comparison["negative_categories"].items():
            category = categories.setdefault(kind, {"color_only": 0, "cnn": 0})
            category["color_only"] += values["color_only"]
            category["cnn"] += values["cnn"]
    totals["color_comparison"]["negative_categories"] = categories
    totals["targets_met"] = (totals["targets_met"]
                            and totals["direct_return_count"] == len(results)
                            and totals["color_comparison"]["same_color_only_reports"] == 0
                            and totals["color_comparison"]["color_only_false_candidates"] > 0
                            and totals["color_comparison"]["cnn_false_candidates"]
                            < totals["color_comparison"]["color_only_false_candidates"]
                            and all(not item.get("flight", {}).get("collision", True)
                                    and item["flight"]["viewpoint_count"] == 3 * item["flight"]["survey_count"]
                                    and item["flight"]["exploration_method"] == "cnn_triggered"
                                    and item["flight"]["fixed_capture_count"] == 0
                                    and item["flight"]["survey_count"] == item["marker_count"]
                                    for item in results))
    if layout_variations:
        totals["targets_met"] = (totals["targets_met"]
                                and totals["matched_count"] == totals["marker_count"]
                                and totals["false_reports"] == 0)
    (directory / "summary.json").write_text(json.dumps(totals, indent=2), encoding="utf-8")
    print(json.dumps(totals, indent=2), flush=True)
    if not totals["targets_met"]:
        raise SystemExit("Measured targets were not all met; inspect results and do not claim success")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/evaluation")
    parser.add_argument("--layout-variations", action="store_true")
    args = parser.parse_args()
    run_suite(args.output, args.layout_variations)
