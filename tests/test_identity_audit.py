import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from identity_audit import audit_identity, load_identity_audit, normalize_addresses
from data_preparation import MONTHS


class IdentityAuditTests(unittest.TestCase):
    def test_address_format_is_normalized_without_inventing_missing_identities(self):
        observed = normalize_addresses(pd.Series([" 0x4CA80D ", "4ca80d", "unknown", None, "12345", "0xGGGGGG"]))
        self.assertEqual(observed.iloc[:2].tolist(), ["4ca80d", "4ca80d"])
        self.assertTrue(observed.iloc[2:].isna().all())

    def test_chunked_identity_audit_resolves_calendar_overlap_and_checks_feature_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, features, output = root / "raw", root / "features", root / "audit"
            raw.mkdir(); features.mkdir()
            for name, month in MONTHS.items():
                features.joinpath(f"features_{name}.csv").write_text("synthetic fixture\n", encoding="utf-8")
                rows = [{"my_id": f"02-{name[:3].title()}-2021_SINGLE", "mode_s": "0x123456"}]
                if month == 1:
                    rows += [{"my_id": "02-Jan-2021_SHARED", "mode_s": value}
                             for value in ["0x4CA80D", "4ca80d", "0x4CA226", "missing"]]
                    rows.append({"my_id": "02-Feb-2021_OWNER", "mode_s": "0xaaaaaa"})
                if month == 2:
                    rows.append({"my_id": "02-Feb-2021_OWNER", "mode_s": "0xbbbbbb"})
                pd.DataFrame(rows).to_csv(raw / f"cleanedx_{month:02}.csv", index=False)
            summary = audit_identity(raw, features, output, chunksize=2)
            self.assertEqual(summary["ambiguous_daily_ids"], ["02-Jan-2021_SHARED"])
            self.assertFalse(summary["target_used_for_rule"])
            self.assertEqual(load_identity_audit(output / "identity_audit.json", features), {"02-Jan-2021_SHARED"})
            (features / "features_jan.csv").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Feature input changed"):
                load_identity_audit(output / "identity_audit.json", features)


if __name__ == "__main__":
    unittest.main()
