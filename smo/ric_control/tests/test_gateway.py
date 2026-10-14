"""Application and restoration must come from simulator receipts, not send success."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from smo.ric_control import state


class ReconciliationTests(unittest.TestCase):
    def reconcile(self, events, alive, status="submitted"):
        ident = "a" * 32
        run = "b" * 32
        saved = {
            "deployment_id": ident,
            "live_run_id": run,
            "status": status,
            "forwarded": True,
            "applied": False,
            "restored": False,
            "receipts": [],
            "ns3_application_confirmed": False,
        }
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / run
            directory.mkdir()
            text = "".join(
                json.dumps(dict(command_id=ident, run_id=run, status=event)) + "\n"
                for event in events
            )
            (directory / "receipts.jsonl").write_text(text + '{"incomplete":')
            with (
                patch.object(state, "RUNS", Path(temp)),
                patch.object(state, "read_saved", return_value=copy.deepcopy(saved)),
                patch.object(state, "live_run", return_value={"process_alive": alive}),
            ):
                return state.reconcile(ident)

    def test_sent_is_not_applied(self):
        result = self.reconcile([], True)
        self.assertEqual(result["status"], "submitted")
        self.assertFalse(result["ns3_application_confirmed"])

    def test_stopped_before_start_is_unconfirmed(self):
        result = self.reconcile(["scheduled"], False)
        self.assertEqual(result["status"], "unconfirmed_simulation_stopped")
        self.assertFalse(result["applied"])

    def test_stopped_after_application_does_not_claim_restoration(self):
        result = self.reconcile(["scheduled", "applied"], False)
        self.assertEqual(result["status"], "restore_unconfirmed_simulation_stopped")
        self.assertTrue(result["applied"])
        self.assertFalse(result["restored"])

    def test_finished_receipts_survive_partial_last_line(self):
        result = self.reconcile(["scheduled", "applied", "restored", "duplicate_ignored"], False)
        self.assertEqual(result["status"], "restored")
        self.assertTrue(result["applied"])
        self.assertTrue(result["restored"])
