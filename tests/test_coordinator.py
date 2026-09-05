import unittest

from lamp_core.coordinator import AppEvent
from simulate_system import build


class CoordinatorTests(unittest.TestCase):
    def test_normal_stable_page_moves_and_speaks(self):
        controller, bus, speaker = build()
        controller.handle(AppEvent.BOOK_MOVED)
        controller.handle(AppEvent.BOOK_STILL)
        self.assertEqual(controller.safety.state.name, "READY_HOLD")
        self.assertGreater(len(bus.received_frames), 1)
        self.assertEqual(len(speaker.messages), 1)

    def test_bus_timeout_faults_and_disables_drives(self):
        controller, bus, speaker = build()
        bus.disconnect()
        controller.handle(AppEvent.BOOK_STILL)
        self.assertEqual(controller.safety.state.name, "FAULT_LATCHED")
        self.assertFalse(controller.safety.drives_enabled)
        self.assertEqual(speaker.messages, [])

    def test_limit_faults_without_motion_frames(self):
        controller, bus, _ = build()
        bus.trigger_limit("j2_shoulder")
        controller.handle(AppEvent.BOOK_STILL)
        self.assertEqual(controller.safety.state.name, "FAULT_LATCHED")
        self.assertEqual(bus.received_frames, [])

    def test_estop_blocks_later_vision_event(self):
        controller, bus, _ = build()
        controller.handle(AppEvent.ESTOP)
        controller.handle(AppEvent.BOOK_STILL)
        self.assertEqual(controller.safety.state.name, "SAFE_DISABLED")
        self.assertEqual(bus.received_frames, [])
