from pathlib import Path
import tempfile
import unittest
from tools.check_docs import check
from tools.check_depth import prose_without_fences, cjk_count
from tools.normalize_math import normalize


class DocumentationChecks(unittest.TestCase):
    def test_inline_calls_and_fenced_examples_are_not_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lesson.md").write_text(
                "Call `checks[typ](value)` here.\n\n"
                "~~~python\ntext = '[example](not-a-file)'\n~~~\n", encoding="utf-8")
            self.assertEqual(check(root), [])

    def test_real_missing_link_is_still_reported_with_code_in_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lesson.md").write_text("[`missing.py`](missing.py)\n", encoding="utf-8")
            errors = check(root)
            self.assertEqual(len(errors), 1)
            self.assertIn("missing.py", errors[0])

    def test_longer_fences_do_not_close_on_shorter_embedded_fence(self):
        text = "正文\n````markdown\n```python\n不能计数\n```\n````\n结尾\n"
        self.assertEqual(cjk_count(prose_without_fences(text)), 4)

    def test_math_normalization_changes_prose_only_and_is_idempotent(self):
        source = "Math \\(x^2\\).\n\\[\na+b\n\\]\n`\\(literal\\)`\n"
        expected = "Math $x^2$.\n$$\na+b\n$$\n`\\(literal\\)`\n"
        self.assertEqual(normalize(source), expected)
        self.assertEqual(normalize(expected), expected)

    def test_math_normalization_preserves_nested_fences_and_inline_code(self):
        source = "````markdown\n```python\n\\(literal\\)\n```\n````\n"
        source += "``one ` tick \\(literal\\)`` and \\(x\\)\n"
        self.assertEqual(normalize(source), source.replace("and \\(x\\)", "and $x$"))


if __name__ == "__main__":
    unittest.main()
