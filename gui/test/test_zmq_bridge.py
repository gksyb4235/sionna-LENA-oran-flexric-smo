"""Compatibility tests for the external live-state GUI bridge."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sionna_rt_gui.zmq_bridge import ZMQBridge


class FakeRadioDevice:
    def __init__(self, position):
        self.position = position


class FakeScene:
    def __init__(self):
        self.receivers = {"manual_rx": FakeRadioDevice([9, 9, 1])}
        self.transmitters = {}

    def get(self, name):
        return self.receivers.get(name) or self.transmitters.get(name)

    def remove(self, name):
        self.receivers.pop(name, None)
        self.transmitters.pop(name, None)


class FakeGui:
    def __init__(self):
        self.scene = FakeScene()
        self.cfg = SimpleNamespace(
            paths=SimpleNamespace(auto_update=False)
        )
        self.antenna_updates = []

    def add_radio_device(
        self,
        position,
        *,
        is_transmitter,
        allow_auto_update,
        name,
    ):
        devices = (
            self.scene.transmitters
            if is_transmitter
            else self.scene.receivers
        )
        devices[name] = FakeRadioDevice(position)

    def set_gnb_antenna_state(self, name, bearing_deg, tilt_deg, num_v):
        self.antenna_updates.append(
            (name, bearing_deg, tilt_deg, num_v)
        )


class ZmqBridgeStateTest(unittest.TestCase):
    @patch(
        "sionna_rt_gui.sionna_utils."
        "set_or_update_radio_devices_polyscope"
    )
    def test_batch_updates_active_set_without_removing_manual_receivers(
        self,
        _update_polyscope,
    ):
        bridge = ZMQBridge()
        gui = FakeGui()

        bridge._apply_simulation_state(
            gui,
            {
                "ues": [
                    {"name": "ue1", "position": [1, 2, 1.5]},
                    {"name": "ue2", "position": [3, 4, 1.5]},
                ],
                "gnbs": [
                    {
                        "name": "gnb1",
                        "position": [0, 0, 30],
                        "bearing_deg": 0,
                        "tilt_deg": 6,
                        "num_v": 1,
                    }
                ],
            },
        )
        bridge._apply_simulation_state(
            gui,
            {
                "ues": [
                    {"name": "ue2", "position": [5, 6, 1.5]},
                ],
                "gnbs": [
                    {
                        "name": "gnb1",
                        "position": [0, 0, 30],
                        "bearing_deg": 0,
                        "tilt_deg": 6,
                        "num_v": 1,
                    }
                ],
            },
        )

        self.assertNotIn("ue1", gui.scene.receivers)
        self.assertEqual(gui.scene.receivers["ue2"].position, [5, 6, 1.5])
        self.assertIn("manual_rx", gui.scene.receivers)
        self.assertEqual(
            gui.antenna_updates,
            [("gnb1", 0.0, 6.0, 1)],
        )


if __name__ == "__main__":
    unittest.main()
