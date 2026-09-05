"""Run a safe regression profile for the picture-book lamp.

This is an orchestrator, not a second implementation: every stage invokes the
same production modules and existing test entry points used by the app.  It
never opens GPIO, serial, CAN, or real motors.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent


COMPONENT_TEST_PATTERNS: dict[str, tuple[str, ...]] = {
    "reading": ("test_cloud.py", "test_reading_prompt.py", "test_vision.py"),
    "safety": ("test_safety.py", "test_coordinator.py", "test_behavior.py", "test_smart_motion.py"),
    "motion": ("test_motion.py", "test_ideal_plant.py", "test_action_motion.py", "test_mechanicalarm_port.py"),
    "can": ("test_mks_can_protocol.py", "test_module_communication.py"),
    "autonomous": ("test_autonomous*.py", "test_event_router.py", "test_model_router.py"),
}


@dataclass
class StageResult:
    name: str
    command: list[str]
    passed: bool
    duration_s: float
    missing_expectations: list[str]


def run_stage(name: str, command: Sequence[str], expected: Sequence[str] = ()) -> StageResult:
    """Execute an existing program and verify its observable contract."""

    print(f"\n=== {name} ===")
    started = time.monotonic()
    completed = subprocess.run(
        list(command),
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    duration_s = time.monotonic() - started
    print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    missing = [text for text in expected if text not in completed.stdout]
    passed = completed.returncode == 0 and not missing
    if not passed:
        print(f"FAILED: returncode={completed.returncode}; missing={missing}")
    return StageResult(name, list(command), passed, duration_s, missing)


def write_report(path: Path, profile: str, stages: list[StageResult]) -> None:
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "profile": profile,
        "passed": all(stage.passed for stage in stages),
        "stages": [asdict(stage) for stage in stages],
        "scope": (
            "Software-only regression. No GPIO, serial, CAN or real motors are opened. "
            "Cloud image/TTS is included only when --with-cloud-image is supplied."
        ),
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("quick", "full"),
        default="quick",
        help="quick is for every update; full adds exhaustive motion and all scenario scans",
    )
    parser.add_argument(
        "--component",
        choices=tuple((*COMPONENT_TEST_PATTERNS, "system")),
        help="run only the tests for one changed subsystem; use this for normal development",
    )
    parser.add_argument(
        "--with-cloud-image",
        type=Path,
        help="run the real JPEG -> cloud vision -> TTS-WAV test using this image",
    )
    parser.add_argument("--report", type=Path, default=Path("full-test-report.json"))
    args = parser.parse_args()
    if args.component and args.profile == "full":
        parser.error("--component and --profile full cannot be combined; full always scans the whole project")

    python = sys.executable
    if args.component:
        if args.component == "system":
            stages = [
                ("system scenario: normal", [python, "simulate_system.py", "normal"], ("FINAL_SAFETY_STATE READY_HOLD",)),
                ("system scenario: bus timeout", [python, "simulate_system.py", "timeout"], ("FINAL_SAFETY_STATE FAULT_LATCHED",)),
                ("system scenario: hardware limit", [python, "simulate_system.py", "limit"], ("FINAL_SAFETY_STATE FAULT_LATCHED",)),
                ("system scenario: camera loss", [python, "simulate_system.py", "camera"], ("FINAL_SAFETY_STATE FAULT_LATCHED",)),
                ("system scenario: emergency stop", [python, "simulate_system.py", "estop"], ("FINAL_SAFETY_STATE SAFE_DISABLED",)),
                ("production main offline flow", [python, "app_main.py", "--mode", "simulate"], ("READY:",)),
            ]
        else:
            stages = [
                (
                    f"{args.component} contract tests: {pattern}",
                    [python, "-m", "unittest", "discover", "-s", "tests", "-p", pattern, "-v"],
                    ("OK",),
                )
                for pattern in COMPONENT_TEST_PATTERNS[args.component]
            ]
            if args.component == "motion":
                stages.append(
                    ("representative ideal motion profiles", [python, "simulate_ideal_verification.py"], ("PASS:",))
                )
            if args.component == "reading":
                stages.append(
                    ("production main offline flow", [python, "app_main.py", "--mode", "simulate"], ("READY:",))
                )
    else:
        stages = [
            ("unit and contract tests", [python, "-m", "unittest", "discover", "-s", "tests", "-v"], ("OK",)),
            ("production main offline flow", [python, "app_main.py", "--mode", "simulate"], ("READY:",)),
        ]
    if args.profile == "full":
        stages.extend([
        (
            "ideal trajectory and action-catalogue sweep",
            [python, "simulate_ideal_verification.py", "--exhaustive", "--catalogue"],
            ("CORNER_SWEEP transitions=992", "PASS:"),
        ),
        ("system scenario: normal", [python, "simulate_system.py", "normal"], ("FINAL_SAFETY_STATE READY_HOLD",)),
        ("system scenario: bus timeout", [python, "simulate_system.py", "timeout"], ("FINAL_SAFETY_STATE FAULT_LATCHED",)),
        ("system scenario: hardware limit", [python, "simulate_system.py", "limit"], ("FINAL_SAFETY_STATE FAULT_LATCHED",)),
        ("system scenario: camera loss", [python, "simulate_system.py", "camera"], ("FINAL_SAFETY_STATE FAULT_LATCHED",)),
        ("system scenario: emergency stop", [python, "simulate_system.py", "estop"], ("FINAL_SAFETY_STATE SAFE_DISABLED",)),
        ])
    if args.with_cloud_image:
        stages.append(
            (
                "production JPEG/cloud/TTS substitution flow",
                [python, "app_main_test.py", str(args.with_cloud_image)],
                ("Picture-book page accepted:", "WAV written:", "Recognition result written:"),
            )
        )

    results: list[StageResult] = []
    try:
        for name, command, expected in stages:
            result = run_stage(name, command, expected)
            results.append(result)
            if not result.passed:
                break
    finally:
        report_profile = args.component or args.profile
        write_report(args.report, report_profile, results)
        print(f"\nReport written: {args.report}")

    if len(results) != len(stages) or not all(result.passed for result in results):
        print("FULL REGRESSION: FAILED")
        return 1
    print("FULL REGRESSION: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
