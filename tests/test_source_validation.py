import unittest

import pandas as pd

from src.validate_sessions import classify_interval, clock_hypotheses, split_clock_sessions


class SourceValidationTests(unittest.TestCase):
    def test_session_gap_is_strictly_greater_than_provenance_threshold(self):
        self.assertEqual(len(split_clock_sessions([0, 2400])), 1)
        self.assertEqual(len(split_clock_sessions([0, 2401])), 2)

    def test_clock_hypotheses_differ_by_season(self):
        summer = pd.Timestamp("2021-07-01T10:00:00Z").timestamp()
        mappings = clock_hypotheses(summer, summer + 60)
        self.assertEqual(len(mappings), 2)
        self.assertEqual(mappings[0]["start_clock_seconds"] - mappings[1]["start_clock_seconds"], -3600)
        winter = pd.Timestamp("2021-01-01T10:00:00Z").timestamp()
        winter_mappings = clock_hypotheses(winter, winter + 60)
        self.assertEqual(winter_mappings[0]["start_clock_seconds"], winter_mappings[1]["start_clock_seconds"])

    def test_single_london_clock_session_supports_matching_interval(self):
        start = pd.Timestamp("2021-07-01T10:00:00Z").timestamp()
        raw_start = pd.Timestamp("2021-07-01 11:00:00").value // 10**9
        raw_end = raw_start + 600
        raw = {"times": [raw_start, raw_end], "addresses": {"ABC123"},
               "invalid_clocks": 0, "invalid_addresses": 0}
        snapped = pd.DataFrame([{"index": 3, "t1": start, "t2": start + 600}])
        result, sessions = classify_interval(10.0, raw, snapped)
        self.assertEqual(result["status"], "supported_observation_session")
        self.assertEqual(result["mapping_names"], "Europe_London_clock")
        self.assertEqual(len(sessions), 1)

    def test_missing_valid_aircraft_address_requires_review(self):
        start = pd.Timestamp("2021-01-01T10:00:00Z").timestamp()
        raw = {"times": [int(start), int(start + 600)], "addresses": set(),
               "invalid_clocks": 0, "invalid_addresses": 2}
        snapped = pd.DataFrame([{"index": 3, "t1": start, "t2": start + 600}])
        result, _ = classify_interval(10.0, raw, snapped)
        self.assertEqual(result["status"], "needs_source_review")
        self.assertIn("no_valid_source_aircraft_address", result["flags"])


if __name__ == "__main__":
    unittest.main()
