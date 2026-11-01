import json
from pathlib import Path
import tempfile
import unittest
from tools.course import catalog, eligible, prepare_command, read_progress, record, safe_path, validate


class CourseToolsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for tid in "FARMSK":
            book = self.root / "books" / tid
            book.mkdir(parents=True)
            (book / "lesson.md").write_text("# Lesson", encoding="utf-8")
            data = {"track_id": tid, "title": tid, "chapters": [{"id": tid + "01", "title": "Lesson", "path": f"books/{tid}/lesson.md", "prerequisites": [], "labs": [{"id": tid + "01-L1", "command": "python -c pass", "hardware": "cpu", "network": False}]}]}
            (book / "manifest.json").write_text(json.dumps(data), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_catalog(self):
        self.assertEqual(validate(self.root), [])
        self.assertEqual(len(catalog(self.root)[1]), 6)

    def test_missing_path_is_detected(self):
        (self.root / "books/F/lesson.md").unlink()
        self.assertTrue(any("missing chapter" in x for x in validate(self.root)))

    def test_escape_rejected(self):
        with self.assertRaises(ValueError):
            safe_path(self.root, "../outside")

    def test_progress_durable_and_replacement(self):
        record("F01", "practiced", "ran variants", self.root)
        record("F01", "mastered", "explained invariant", self.root)
        self.assertEqual(read_progress(self.root)["chapters"]["F01"]["status"], "mastered")
        self.assertEqual(len(list((self.root / ".course").glob("*.tmp"))), 0)

    def test_record_rejects_unknown_and_empty_evidence(self):
        with self.assertRaises(ValueError):
            record("BAD", "reading", "text", self.root)
        with self.assertRaises(ValueError):
            record("F01", "mastered", " ", self.root)

    def test_practiced_does_not_unlock_mastery_gate(self):
        chapters = [{"id": "A01", "prerequisites": ["F01"]}]
        self.assertEqual(eligible(chapters, {"chapters": {"F01": {"status": "practiced"}}}), [])
        self.assertEqual(len(eligible(chapters, {"chapters": {"F01": {"status": "mastered"}}})), 1)

    def test_network_and_gpu_require_explicit_optin(self):
        lab = {"command": "python -m example", "hardware": "cuda", "network": True}
        with self.assertRaises(ValueError):
            prepare_command(lab, False, False)
        with self.assertRaises(ValueError):
            prepare_command(lab, False, True)
        self.assertEqual(prepare_command(lab, True, True)[1:], ["-m", "example"])

    def test_shell_not_interpreted(self):
        with self.assertRaises(ValueError):
            prepare_command({"command": "python x.py && rm file"}, False, False)

    def test_cycle_detected(self):
        p = self.root / "books/F/manifest.json"
        data = json.loads(p.read_text())
        data["chapters"][0]["prerequisites"] = ["F01"]
        p.write_text(json.dumps(data))
        self.assertTrue(any("cycle" in x for x in validate(self.root)))


if __name__ == "__main__":
    unittest.main()
