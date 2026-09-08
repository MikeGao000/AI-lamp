"""Regression tests for the user-specified MuJoCo lamp geometry.

These tests intentionally parse XML only: they remain runnable in the standard
test environment and do not require the optional MuJoCo viewer dependency.
"""

from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


MODEL_PATH = Path(__file__).parents[1] / "simulations" / "mujoco" / "lamp_5axis.xml"


def named_element(root: ET.Element, name: str) -> ET.Element:
    element = root.find(f".//*[@name='{name}']")
    if element is None:
        raise AssertionError(f"missing element: {name}")
    return element


class MuJoCoGeometrySpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = ET.parse(MODEL_PATH).getroot()

    def test_axis_chain_has_the_specified_centre_to_centre_lengths(self):
        j2 = named_element(self.root, "j2_shoulder")
        j3 = named_element(self.root, "j3_elbow")
        j4 = named_element(self.root, "j4_neck")
        j5 = named_element(self.root, "j5_head")

        self.assertEqual("0 0 0.05", j2.attrib["pos"])
        self.assertEqual("0.20 0 0", j3.attrib["pos"])
        self.assertEqual("0.09 0 0", j4.attrib["pos"])
        self.assertEqual("0 0 0.05", j5.attrib["pos"])
        large_arm = named_element(self.root, "large_arm_200mm")
        small_arm = named_element(self.root, "small_arm_90mm")
        self.assertEqual("box", large_arm.attrib["type"])
        self.assertEqual("0.10 0.015 0.018", large_arm.attrib["size"])
        self.assertEqual("box", small_arm.attrib["type"])
        self.assertEqual("0.045 0.014 0.016", small_arm.attrib["size"])

    def test_each_joint_uses_a_42_mm_stepper_motor_envelope(self):
        expected_half_sizes = {
            "j1_motor": "0.021 0.021 0.020",
            "j2_motor": "0.021 0.020 0.021",
            "j3_motor": "0.021 0.020 0.021",
            "j4_motor": "0.021 0.020 0.021",
            "j5_motor": "0.021 0.021 0.020",
        }
        for motor_name, expected_size in expected_half_sizes.items():
            motor = named_element(self.root, motor_name)
            self.assertEqual("box", motor.attrib["type"])
            self.assertEqual(expected_size, motor.attrib["size"])

    def test_u_arms_are_plate_frames_between_the_green_motor_positions(self):
        # The real red members are solid U-arm frames, not thin wire capsules.
        self.assertEqual(
            "box",
            named_element(self.root, "base_u_bridge").attrib["type"],
        )
        self.assertEqual(
            "box",
            named_element(self.root, "neck_u_bridge").attrib["type"],
        )

    def test_lamp_head_is_proportional_to_a_42_mm_stepper(self):
        head = named_element(self.root, "head_shell")
        self.assertEqual("0.065 0.050 0.040", head.attrib["size"])
        self.assertLessEqual(2 * float(head.attrib["size"].split()[1]), 0.105)

    def test_drag_sliders_are_bounded_by_existing_joint_ranges(self):
        expected_ranges = {
            "j1_position": "-1.57 1.57",
            "j2_position": "-0.78 0.78",
            "j3_position": "-0.95 0.95",
            "j4_position": "-0.70 0.70",
            "j5_position": "-1.05 1.05",
        }
        for actuator_name, expected_range in expected_ranges.items():
            actuator = named_element(self.root, actuator_name)
            self.assertEqual("true", actuator.attrib["ctrllimited"])
            self.assertEqual(expected_range, actuator.attrib["ctrlrange"])


if __name__ == "__main__":
    unittest.main()
