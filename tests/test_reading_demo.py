import unittest

from lamp_core.camera_kinematics import DEFAULT_IDEAL_CAMERA_MODEL
from lamp_core.cloud import StaticStoryClient
from lamp_core.reading_demo import (
    READING_POSTURE_RAD,
    ReadingDemoError,
    SimulatedBook,
    center_book_in_virtual_frame,
    gaze_action_for_direction,
    _prepared_runtime,
    run_simulated_book_reading_flow,
)
from simulate import JOINT_LIMITS
from lamp_core.virtual_hardware import VirtualMotorBus


class ReadingDemoTests(unittest.TestCase):
    def test_target_is_reprojected_after_all_camera_joints_move(self):
        book = SimulatedBook("book", (0.35, 0.35, 0.65, 0.65))
        initial = book.observation(camera_joint_positions_rad={name: 0.0 for name in DEFAULT_IDEAL_CAMERA_MODEL.joint_names})
        moved = book.observation(
            camera_joint_positions_rad={
                "j1_base_yaw": 0.12,
                "j2_shoulder": -0.08,
                "j3_elbow": 0.09,
                "j4_neck_pitch": -0.06,
                "j5_head_yaw": 0.11,
            }
        )
        initial_box = initial.objects[0].bbox_norm
        moved_box = moved.objects[0].bbox_norm
        self.assertNotEqual(initial_box, moved_box)
        self.assertNotEqual(
            (initial_box[0] + initial_box[2]) / 2.0,
            (moved_box[0] + moved_box[2]) / 2.0,
        )
        self.assertNotEqual(
            (initial_box[1] + initial_box[3]) / 2.0,
            (moved_box[1] + moved_box[3]) / 2.0,
        )

    def test_book_at_x_018_is_visually_centered_before_capture_and_reading(self):
        result = run_simulated_book_reading_flow(
            SimulatedBook("book-left", (0.08, 0.20, 0.32, 0.88)),
            StaticStoryClient("Det er en lille bjørn med en bog."),
            JOINT_LIMITS,
        )
        self.assertEqual("left", result.book_direction.value)
        self.assertEqual("VISUAL_SERVO_CENTER", result.gaze_action_id)
        self.assertEqual(
            (
                "book_found",
                "book_centered",
                "snapshot_captured",
                "book_recognized",
                "reading_posture",
                "reading_spoken",
            ),
            result.flow_steps,
        )
        self.assertAlmostEqual(0.5, result.centered_book_x, delta=0.04)
        self.assertAlmostEqual(0.5, result.centered_book_y, delta=0.04)
        self.assertEqual(1, result.visual_servo_steps)
        self.assertTrue(all(abs(value) > 1e-6 for value in result.centering_positions_rad.values()))
        for joint_name, expected in READING_POSTURE_RAD.items():
            self.assertAlmostEqual(expected, result.final_positions_rad[joint_name])
        self.assertGreater(result.motor_frame_count, 2)
        self.assertEqual(("Det er en lille bjørn med en bog.",), result.spoken_messages)

    def test_book_sector_selects_the_expected_gaze_action(self):
        self.assertEqual("LOOK_LEFT", gaze_action_for_direction(SimulatedBook("left", (0.0, 0.2, 0.2, 0.8)).direction))
        self.assertEqual("LOOK_CENTER", gaze_action_for_direction(SimulatedBook("center", (0.4, 0.2, 0.6, 0.8)).direction))
        self.assertEqual("LOOK_RIGHT", gaze_action_for_direction(SimulatedBook("right", (0.75, 0.2, 0.95, 0.8)).direction))

    def test_right_book_is_centered_by_opposite_j1_motion(self):
        result = run_simulated_book_reading_flow(
            SimulatedBook("book-right", (0.72, 0.20, 0.92, 0.88)),
            StaticStoryClient("A book on the right."),
            JOINT_LIMITS,
        )
        self.assertEqual("right", result.book_direction.value)
        self.assertAlmostEqual(0.5, result.centered_book_x, delta=0.04)
        self.assertAlmostEqual(0.5, result.centered_book_y, delta=0.04)
        self.assertGreater(result.visual_servo_steps, 0)

    def test_centered_book_skips_unnecessary_servo_motion(self):
        result = run_simulated_book_reading_flow(
            SimulatedBook("book-center", (0.40, 0.20, 0.60, 0.80)),
            StaticStoryClient("A centered book."),
            JOINT_LIMITS,
        )
        self.assertEqual(0, result.visual_servo_steps)
        self.assertAlmostEqual(0.5, result.centered_book_x)
        self.assertAlmostEqual(0.5, result.centered_book_y)

    def test_lower_book_is_centered_with_j4_pitch_correction(self):
        result = run_simulated_book_reading_flow(
            SimulatedBook("book-lower", (0.40, 0.78, 0.60, 0.94)),
            StaticStoryClient("A lower book."),
            JOINT_LIMITS,
        )
        self.assertAlmostEqual(0.5, result.centered_book_x, delta=0.04)
        self.assertAlmostEqual(0.5, result.centered_book_y, delta=0.04)
        self.assertGreater(result.visual_servo_steps, 0)

    def test_lower_edge_target_can_use_the_existing_j4_limit_exactly(self):
        result = run_simulated_book_reading_flow(
            SimulatedBook("book-lower-edge", (0.05, 0.83, 0.56, 0.99)),
            StaticStoryClient("A lower edge book."),
            JOINT_LIMITS,
        )
        self.assertAlmostEqual(0.5, result.centered_book_x, delta=0.04)
        self.assertAlmostEqual(0.5, result.centered_book_y, delta=0.04)

    def test_unreachable_centering_configuration_reports_failure(self):
        with self.assertRaises(ReadingDemoError):
            center_book_in_virtual_frame(
                SimulatedBook("book-left", (0.08, 0.20, 0.32, 0.88)),
                _prepared_runtime(),
                VirtualMotorBus(JOINT_LIMITS),
                "test-centering",
                max_iterations=1,
            )
