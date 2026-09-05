"""Run normal and fault scenarios for the reading-companion main coordinator."""

from __future__ import annotations

import argparse
import sys

from lamp_core.coordinator import AppEvent, ReadingCompanionCoordinator
from lamp_core.motion import JointLimit
from lamp_core.virtual_hardware import SpeechStub, VirtualMotorBus


LIMITS = {
    "j1_base_yaw": JointLimit(-1.57, 1.57, 0.80),
    "j2_shoulder": JointLimit(-0.78, 0.78, 0.45),
    "j3_elbow": JointLimit(-0.95, 0.95, 0.55),
    "j4_neck_pitch": JointLimit(-0.70, 0.70, 0.75),
    "j5_head_yaw": JointLimit(-1.05, 1.05, 0.90),
}


def build() -> tuple[ReadingCompanionCoordinator, VirtualMotorBus, SpeechStub]:
    bus = VirtualMotorBus(LIMITS)
    speaker = SpeechStub()
    controller = ReadingCompanionCoordinator(LIMITS, bus, speaker)
    controller.home()
    return controller, bus, speaker


def run(scenario: str) -> None:
    controller, bus, speaker = build()
    if scenario == "normal":
        controller.handle(AppEvent.BOOK_MOVED)
        controller.handle(AppEvent.BOOK_STILL)
    elif scenario == "timeout":
        bus.disconnect()
        controller.handle(AppEvent.BOOK_STILL)
    elif scenario == "limit":
        bus.trigger_limit("j2_shoulder")
        controller.handle(AppEvent.BOOK_STILL)
    elif scenario == "camera":
        controller.handle(AppEvent.CAMERA_LOST)
        controller.handle(AppEvent.BOOK_STILL)
    elif scenario == "estop":
        controller.handle(AppEvent.ESTOP)
        controller.handle(AppEvent.BOOK_STILL)
    else:
        raise ValueError(scenario)

    print(f"SCENARIO {scenario.upper()}")
    for line in controller.log:
        print(line)
    print(f"FINAL_SAFETY_STATE {controller.safety.state.name}")
    print(f"VIRTUAL_FRAMES {len(bus.received_frames)}")
    for message in speaker.messages:
        print(f"SPEECH {message}")


if __name__ == "__main__":
    # Some Windows shells still expose a legacy code page. Keep simulations
    # observable there instead of failing while rendering a Chinese TTS stub.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=("normal", "timeout", "limit", "camera", "estop"))
    run(parser.parse_args().scenario)
