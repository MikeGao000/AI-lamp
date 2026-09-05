"""Event-driven coordinator shared by the app simulator and future hardware app."""

from __future__ import annotations

from enum import Enum, auto
from typing import Mapping

from lamp_core.motion import JointLimit, plan_synchronised_minimum_jerk
from lamp_core.safety import MotionSafetyCheck, SafetyController, SafetyError, SafetyState
from lamp_core.virtual_hardware import HardwareLimitError, MotorBusTimeoutError, SpeechStub, VirtualMotorBus


class AppEvent(Enum):
    BOOK_MOVED = auto()
    BOOK_STILL = auto()
    CAMERA_LOST = auto()
    ESTOP = auto()


class ReadingCompanionCoordinator:
    """Coordinates safe reading events without knowing about serial or camera APIs."""

    def __init__(
        self,
        limits: Mapping[str, JointLimit],
        motor_bus: VirtualMotorBus,
        speaker: SpeechStub,
    ) -> None:
        self.limits = limits
        self.bus = motor_bus
        self.speaker = speaker
        self.safety = SafetyController()
        self.log: list[str] = []
        self.pending_story = "我看到新的一页了，我们继续读。"

    def set_pending_story(self, story: str) -> None:
        cleaned = story.strip()
        if not cleaned:
            raise ValueError("story must not be empty")
        self.pending_story = cleaned

    def home(self) -> None:
        self.safety.begin_homing()
        self.log.append("HOMING: simulated normally-closed limits verified")
        self.safety.complete_homing()
        self.safety.enable_drives()
        self.log.append("READY: homing complete; virtual drives enabled")

    def handle(self, event: AppEvent) -> None:
        if event is AppEvent.ESTOP:
            self.safety.emergency_stop("simulated physical E-stop")
            self.log.append("ESTOP: virtual drives disabled immediately")
            return
        if event is AppEvent.CAMERA_LOST:
            self.safety.fault("camera stream lost")
            self.log.append("FAULT: camera lost; motion inhibited")
            return
        if self.safety.state is not SafetyState.READY_HOLD:
            self.log.append(f"IGNORED: {event.name} while state is {self.safety.state.name}")
            return
        if event is AppEvent.BOOK_MOVED:
            self.log.append("VISION: book moved; stability timer reset")
            return
        if event is AppEvent.BOOK_STILL:
            self._read_page()

    def _read_page(self) -> None:
        target = {
            "j1_base_yaw": 0.25,
            "j2_shoulder": -0.20,
            "j3_elbow": 0.30,
            "j4_neck_pitch": 0.12,
            "j5_head_yaw": -0.18,
        }
        trajectory = plan_synchronised_minimum_jerk(
            self.bus.positions_rad, target, self.limits, sample_period_s=0.02
        )
        try:
            self.safety.begin_motion(MotionSafetyCheck.simulated_clear())
            self.log.append("VISION: page stable; capture accepted")
            self.bus.execute(trajectory)
            self.safety.finish_motion()
            self.speaker.speak(self.pending_story)
            self.log.append(f"READY: {len(trajectory)} virtual frames completed; speech queued")
        except (MotorBusTimeoutError, HardwareLimitError, ValueError, SafetyError) as error:
            self.safety.fault(str(error))
            self.log.append(f"FAULT: {error}")
