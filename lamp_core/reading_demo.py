"""End-to-end, no-hardware picture-book reading flow for motor integration tests.

The flow is deliberately concrete: a simulated camera observation finds a
book, directs the lamp's gaze to the book sector, captures a stable image,
obtains a story response, moves into the existing reading posture, and sends
the text to a local speech stub.  It uses the same Autonomous runtime and
minimum-jerk five-axis trajectories as the product path, but only a virtual
motor bus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from lamp_core.action_motion import compile_action_to_ideal_segments
from lamp_core.autonomous_contracts import (
    Direction,
    EventKind,
    EventPriority,
    EventSource,
    LampEvent,
    LampProfile,
    Language,
    PrivacyMode,
    VisualObject,
    VisualObservation,
)
from lamp_core.autonomous_runtime import AutonomousRuntime
from lamp_core.behavior import BehaviorProposal, ScreenTarget
from lamp_core.camera_kinematics import CameraProjectionModel, DEFAULT_IDEAL_CAMERA_MODEL
from lamp_core.cloud import StoryClient
from lamp_core.event_router import EventRouter
from lamp_core.model_router import ModelEndpoint, ModelRouter, ModelTask, RoutingRequest
from lamp_core.motion import JointLimit, TrajectoryPoint, plan_synchronised_minimum_jerk
from lamp_core.pose_library import IDLE_POSE, POSE_LIBRARY
from lamp_core.safety import MotionSafetyCheck, SafetyController
from lamp_core.virtual_hardware import SpeechStub, VirtualMotorBus


class ReadingDemoError(ValueError):
    pass


@dataclass(frozen=True)
class SimulatedBook:
    object_id: str
    bbox_norm: tuple[float, float, float, float]
    snapshot_jpeg: bytes = b"simulated-book-page-jpeg"
    confidence: float = 0.95
    label: str = "book"

    @property
    def center_x(self) -> float:
        return (self.bbox_norm[0] + self.bbox_norm[2]) / 2.0

    @property
    def direction(self) -> Direction:
        if self.center_x < 1.0 / 3.0:
            return Direction.LEFT
        if self.center_x > 2.0 / 3.0:
            return Direction.RIGHT
        return Direction.CENTER

    def observation(
        self,
        *,
        camera_yaw_rad: float = 0.0,
        yaw_to_screen_gain: float = 0.58,
        camera_pitch_rad: float = 0.0,
        pitch_to_screen_gain: float = 0.58,
        camera_joint_positions_rad: Mapping[str, float] | None = None,
        camera_model: CameraProjectionModel = DEFAULT_IDEAL_CAMERA_MODEL,
    ) -> VisualObservation:
        """Return the virtual-camera observation after the lamp has panned.

        ``bbox_norm`` is the book position in the initial camera view.  The
        simulated camera projects a positive J1 pan to the left in image
        coordinates, matching the local LOOK_LEFT / LOOK_RIGHT pose signs.
        The gain is a deliberately explicit simulation calibration, so the
        visual-servo loop can be tested before camera-to-J1 calibration exists.
        """

        if camera_joint_positions_rad is None:
            # Compatibility path for focused J1/J4 unit fixtures.
            camera_joint_positions_rad = {
                name: 0.0 for name in camera_model.joint_names
            }
            camera_joint_positions_rad["j1_base_yaw"] = camera_yaw_rad * yaw_to_screen_gain / 0.50
            camera_joint_positions_rad["j4_neck_pitch"] = camera_pitch_rad * pitch_to_screen_gain / 0.38
        offset_x, offset_y = camera_model.image_offset(camera_joint_positions_rad)
        x1, y1, x2, y2 = self.bbox_norm
        observed_bbox = (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)
        observation = VisualObservation(
            source="simulated_camera",
            frame_age_ms=30,
            direction=_direction_for_x((observed_bbox[0] + observed_bbox[2]) / 2.0),
            objects=(VisualObject(self.object_id, self.label, observed_bbox, self.confidence),),
        )
        observation.validate()
        if not self.snapshot_jpeg:
            raise ReadingDemoError("book snapshot must not be empty")
        return observation

    def screen_target(
        self,
        *,
        camera_yaw_rad: float = 0.0,
        yaw_to_screen_gain: float = 0.58,
        camera_pitch_rad: float = 0.0,
        pitch_to_screen_gain: float = 0.58,
        camera_joint_positions_rad: Mapping[str, float] | None = None,
        camera_model: CameraProjectionModel = DEFAULT_IDEAL_CAMERA_MODEL,
    ) -> ScreenTarget:
        if camera_joint_positions_rad is None:
            camera_joint_positions_rad = {name: 0.0 for name in camera_model.joint_names}
            camera_joint_positions_rad["j1_base_yaw"] = camera_yaw_rad * yaw_to_screen_gain / 0.50
            camera_joint_positions_rad["j4_neck_pitch"] = camera_pitch_rad * pitch_to_screen_gain / 0.38
        offset_x, offset_y = camera_model.image_offset(camera_joint_positions_rad)
        return ScreenTarget(
            object_id=self.object_id,
            label=self.label,
            center_x=self.center_x + offset_x,
            center_y=(self.bbox_norm[1] + self.bbox_norm[3]) / 2.0 + offset_y,
            confidence=self.confidence,
        )


@dataclass(frozen=True)
class ReadingDemoResult:
    trace_id: str
    book_direction: Direction
    gaze_action_id: str
    centered_book_x: float
    centered_book_y: float
    visual_servo_steps: int
    centering_positions_rad: Mapping[str, float]
    recognition_text: str
    final_positions_rad: Mapping[str, float]
    flow_steps: tuple[str, ...]
    motor_frame_count: int
    spoken_messages: tuple[str, ...]


def gaze_action_for_direction(direction: Direction) -> str:
    return {
        Direction.LEFT: "LOOK_LEFT",
        Direction.CENTER: "LOOK_CENTER",
        Direction.RIGHT: "LOOK_RIGHT",
    }.get(direction, "LOOK_CENTER")


def _direction_for_x(center_x: float) -> Direction:
    if center_x < 1.0 / 3.0:
        return Direction.LEFT
    if center_x > 2.0 / 3.0:
        return Direction.RIGHT
    return Direction.CENTER


def _profile() -> LampProfile:
    return LampProfile(
        device_id="picture-lamp-simulation",
        board="virtual_pi_3b",
        joint_names=tuple(IDLE_POSE),
        required_capabilities=frozenset({"motion", "audio", "vision", "system"}),
    )


def _prepared_runtime() -> AutonomousRuntime:
    safety = SafetyController()
    safety.begin_homing()
    safety.complete_homing()
    safety.enable_drives()
    router = ModelRouter(
        (
            ModelEndpoint(
                provider="simulation",
                model="static-reading-vision",
                task=ModelTask.READING_VISION,
                language=None,
                is_local=True,
                supports_vision=True,
                estimated_latency_ms=1,
            ),
        )
    )
    return AutonomousRuntime(_profile(), safety, EventRouter(), router)


def _execute_approved_action(
    runtime: AutonomousRuntime,
    bus: VirtualMotorBus,
    action_id: str,
    trace_id: str,
    now_ms: int,
) -> int:
    compiled = compile_action_to_ideal_segments(action_id, bus.positions_rad, bus.limits)
    for segment in compiled.segments:
        runtime.safety.begin_motion(MotionSafetyCheck.simulated_clear())
        runtime.flow_log.append(trace_id, now_ms, "motion_started", action_id=action_id)
        bus.execute(segment)
        runtime.safety.finish_motion()
        runtime.flow_log.append(trace_id, now_ms, "motion_completed", action_id=action_id)
    return sum(len(segment) for segment in compiled.segments)


def _execute_trajectory(
    runtime: AutonomousRuntime,
    bus: VirtualMotorBus,
    frames: tuple[TrajectoryPoint, ...],
    trace_id: str,
    now_ms: int,
    *,
    motion_name: str,
) -> int:
    """Run one locally calculated trajectory on the same virtual motor path."""

    if not frames:
        return 0
    runtime.safety.begin_motion(MotionSafetyCheck.simulated_clear())
    runtime.flow_log.append(trace_id, now_ms, "motion_started", action_id=motion_name)
    bus.execute(frames)
    runtime.safety.finish_motion()
    runtime.flow_log.append(trace_id, now_ms, "motion_completed", action_id=motion_name)
    return len(frames)


def center_book_in_virtual_frame(
    book: SimulatedBook,
    runtime: AutonomousRuntime,
    bus: VirtualMotorBus,
    trace_id: str,
    *,
    tolerance_norm: float = 0.04,
    camera_model: CameraProjectionModel = DEFAULT_IDEAL_CAMERA_MODEL,
    max_iterations: int = 6,
) -> tuple[VisualObservation, int, int, Mapping[str, float]]:
    """Visual-servo the simulated book into the central camera band.

    Each iteration re-observes the book, distributes horizontal and vertical
    image errors across every calibrated camera joint, and uses the existing
    minimum-jerk planner.
    It deliberately bypasses no motion layer; the only difference from a
    physical test is that ``VirtualMotorBus`` applies the frames.
    """

    if not 0.0 < tolerance_norm < 0.5:
        raise ReadingDemoError("tolerance_norm must be in (0, 0.5)")
    if (
        max_iterations < 1
    ):
        raise ReadingDemoError("visual-servo configuration must be positive")

    total_frames = 0
    for iteration in range(1, max_iterations + 1):
        observation = book.observation(
            camera_joint_positions_rad=bus.positions_rad,
            camera_model=camera_model,
        )
        target = observation.objects[0]
        observed_x = (target.bbox_norm[0] + target.bbox_norm[2]) / 2.0
        observed_y = (target.bbox_norm[1] + target.bbox_norm[3]) / 2.0
        image_error_x = observed_x - 0.5
        image_error_y = observed_y - 0.5
        runtime.flow_log.append(
            trace_id,
            150 + iteration * 10,
            "book_reobserved",
            iteration=iteration,
            book_center_x=round(observed_x, 4),
            book_center_y=round(observed_y, 4),
            image_error_x=round(image_error_x, 4),
            image_error_y=round(image_error_y, 4),
        )
        if abs(image_error_x) <= tolerance_norm + 1e-9 and abs(image_error_y) <= tolerance_norm + 1e-9:
            runtime.flow_log.append(trace_id, 150 + iteration * 10, "book_centered")
            return observation, iteration - 1, total_frames, dict(bus.positions_rad)

        joint_correction = camera_model.minimum_norm_correction(image_error_x, image_error_y)
        target_positions = dict(bus.positions_rad)
        for joint_name, correction in joint_correction.items():
            limit = bus.limits[joint_name]
            target_positions[joint_name] = max(
                limit.minimum_rad,
                min(limit.maximum_rad, target_positions[joint_name] + correction),
            )
        frames = tuple(plan_synchronised_minimum_jerk(bus.positions_rad, target_positions, bus.limits))
        total_frames += _execute_trajectory(
            runtime,
            bus,
            frames,
            trace_id,
            150 + iteration * 10,
            motion_name="visual_servo_yaw",
        )

    raise ReadingDemoError("book did not reach the central camera band")


def run_simulated_book_reading_flow(
    book: SimulatedBook,
    story_client: StoryClient,
    limits: Mapping[str, JointLimit],
    *,
    language: Language = Language.DA_DK,
) -> ReadingDemoResult:
    """Run book found -> gaze -> snapshot -> recognition -> reading -> speech."""

    initial_observation = book.observation()
    runtime = _prepared_runtime()
    bus = VirtualMotorBus(limits)
    speaker = SpeechStub()
    trace_id = f"reading-{book.object_id}"
    steps = ["book_found"]

    event = LampEvent(
        event_id=f"book-stable-{book.object_id}",
        trace_id=trace_id,
        timestamp_ms=100,
        source=EventSource.CAMERA,
        kind=EventKind.BOOK_STABLE,
        priority=EventPriority.INTERACTIVE,
        privacy_mode=PrivacyMode.LOCAL_ONLY,
        language_hint=language,
        payload={"object_id": book.object_id, "direction": initial_observation.direction.value},
    )
    runtime.process_event(
        event,
        now_ms=100,
        model_request=RoutingRequest(
            ModelTask.READING_VISION,
            language,
            PrivacyMode.LOCAL_ONLY,
            requires_vision=True,
        ),
    )

    centered_observation, visual_servo_steps, servo_frames, centering_positions_rad = center_book_in_virtual_frame(
        book, runtime, bus, trace_id
    )
    gaze_action_id = "VISUAL_SERVO_CENTER"
    steps.append("book_centered")

    runtime.flow_log.append(trace_id, 200, "snapshot_captured", object_id=book.object_id)
    steps.append("snapshot_captured")
    recognition_text = story_client.describe_page(book.snapshot_jpeg, prompt="Describe this picture book page.")
    runtime.flow_log.append(trace_id, 250, "book_recognized")
    steps.append("book_recognized")

    runtime.approve_model_behavior(
        BehaviorProposal(
            "READING_POSTURE",
            target_ref=book.object_id,
            intensity=0.3,
            reason="begin reading",
        ),
        {
            book.object_id: book.screen_target(
                camera_joint_positions_rad=centering_positions_rad,
            )
        },
        trace_id=trace_id,
        now_ms=300,
    )
    _execute_approved_action(runtime, bus, "READING_POSTURE", trace_id, 310)
    steps.append("reading_posture")

    speaker.speak(recognition_text)
    runtime.flow_log.append(trace_id, 400, "tts_started")
    steps.append("reading_spoken")
    return ReadingDemoResult(
        trace_id,
        initial_observation.direction,
        gaze_action_id,
        centered_observation.objects[0].bbox_norm[0] / 2.0 + centered_observation.objects[0].bbox_norm[2] / 2.0,
        centered_observation.objects[0].bbox_norm[1] / 2.0 + centered_observation.objects[0].bbox_norm[3] / 2.0,
        visual_servo_steps,
        centering_positions_rad,
        recognition_text,
        dict(bus.positions_rad),
        tuple(steps),
        len(bus.received_frames),
        tuple(speaker.messages),
    )


READING_POSTURE_RAD = POSE_LIBRARY["reading_pose"]
