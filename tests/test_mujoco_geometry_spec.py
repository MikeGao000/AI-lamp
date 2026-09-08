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
        self.assertEqual("0 0 0 0.20 0 0", named_element(self.root, "large_arm_200mm").attrib["fromto"])
        self.assertEqual("0 0 0 0.09 0 0", named_element(self.root, "small_arm_90mm").attrib["fromto"])

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


if __name__ == "__main__":
    unittest.main()
