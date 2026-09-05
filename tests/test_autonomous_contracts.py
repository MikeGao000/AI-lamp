import unittest

from lamp_core.autonomous_contracts import (
    ContractError,
    ConversationSession,
    Direction,
    EventKind,
    EventPriority,
    EventSource,
    LampEvent,
    LampProfile,
    Language,
    MotorMode,
    MotorTelemetry,
    PrivacyMode,
    VisualObject,
    VisualObservation,
)


class AutonomousContractTests(unittest.TestCase):
    def test_valid_five_axis_fail_closed_profile(self):
        profile = LampProfile(
            device_id="picture-lamp-01",
            board="raspberry_pi_3b",
            joint_names=("J1", "J2", "J3", "J4", "J5"),
            required_capabilities=frozenset({"motion", "audio", "vision", "system"}),
            optional_capabilities=frozenset({"tof"}),
        )
        profile.validate()

    def test_motion_profile_cannot_fail_open(self):
        profile = LampProfile(
            device_id="picture-lamp-01",
            board="raspberry_pi_3b",
            joint_names=("J1", "J2", "J3", "J4", "J5"),
            required_capabilities=frozenset({"motion", "audio", "vision", "system"}),
            safety_mode="pass_through",
        )
        with self.assertRaises(ContractError):
            profile.validate()

    def test_visual_observation_is_facts_and_normalized_boxes(self):
        observation = VisualObservation(
            source="cloud_vlm",
            frame_age_ms=120,
            direction=Direction.LEFT,
            objects=(VisualObject("book-1", "book", (0.1, 0.2, 0.7, 0.9), 0.91),),
        )
        observation.validate()

    def test_invalid_visual_box_is_rejected(self):
        observation = VisualObservation(
            source="lan_gpu",
            frame_age_ms=0,
            direction=Direction.UNKNOWN,
            objects=(VisualObject("book-1", "book", (0.7, 0.2, 0.1, 0.9), 0.91),),
        )
        with self.assertRaises(ContractError):
            observation.validate()

    def test_session_has_a_fixed_language_and_provider_set(self):
        session = ConversationSession(
            session_id="s-1",
            language=Language.DA_DK,
            stt_provider="stt-da",
            chat_provider="chat-da",
            tts_provider="tts-da",
            voice_id="da-voice",
            privacy_mode=PrivacyMode.CLOUD_ALLOWED,
        )
        session.validate()

    def test_motor_telemetry_keeps_unknown_fields_as_none(self):
        telemetry = MotorTelemetry(
            joint="J2",
            node_id=2,
            mode=MotorMode.HOLDING,
            target_position_rad=0.3,
            actual_position_rad=0.29,
            velocity_rad_s=0.0,
            following_error_rad=0.01,
            current_or_torque=None,
            temperature_c=None,
            bus_voltage_v=None,
            fault_code=None,
            last_heartbeat_age_ms=40,
        )
        telemetry.validate()

    def test_event_requires_ids_and_nonnegative_time(self):
        event = LampEvent(
            event_id="e-1",
            trace_id="t-1",
            timestamp_ms=1,
            source=EventSource.VOICE,
            kind=EventKind.VOICE_FINAL,
            priority=EventPriority.INTERACTIVE,
            privacy_mode=PrivacyMode.CLOUD_ALLOWED,
            language_hint=Language.ZH_CN,
        )
        event.validate()
