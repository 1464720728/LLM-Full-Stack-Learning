"""Run selected offline examples from the books, without copying their implementations."""
import ast
from contextlib import nullcontext, redirect_stdout
import io
import itertools
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from labs.agents.capstones import coding_project
from labs.agents.contracts import ContractError, Tool, object_schema
from labs.agents.tools import Workspace, arithmetic_tool
from labs.foundations.paths import demo


ROOT = Path(__file__).resolve().parents[1]
F02 = "books/00-foundations/02-files-paths-and-encoding.md"
A08 = "books/01-agent-engineering/08-workspaces.md"
A14 = "books/01-agent-engineering/14-coding-agent.md"
A15 = "books/01-agent-engineering/15-integrations.md"
M07 = "books/03-model-architectures/07-latent-attention.md"
S07 = "books/04-inference-systems/07-sampling-streaming.md"


def python_blocks(relative):
    return re.findall(r"^```python\n(.*?)^```", (ROOT / relative).read_text(encoding="utf-8"), re.M | re.S)


def example(relative, name):
    marker = "# course-example: " + name
    found = [code for code in python_blocks(relative) if marker in code.splitlines()]
    if len(found) != 1:
        raise AssertionError(f"expected one {name} example in {relative}, found {len(found)}")
    return compile(found[0], relative + "#" + name, "exec")


def definition(relative, name, namespace):
    found = [node for code in python_blocks(relative) for node in ast.parse(code).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name]
    if len(found) != 1:
        raise AssertionError(f"expected one {name} definition in {relative}, found {len(found)}")
    tree = ast.fix_missing_locations(ast.Module(body=found, type_ignores=[]))
    exec(compile(tree, relative + "#" + name, "exec"), namespace)
    return namespace[name]


_write_text = Path.write_text


def windows_text_write(path, data, encoding=None, errors=None, newline=None):
    """Simulate Windows default newline translation on either operating system."""
    return _write_text(path, data, encoding=encoding, errors=errors,
                       newline="\r\n" if newline is None else newline)


class TextFileNewlineTests(unittest.TestCase):
    def run_both_modes(self, operation):
        for simulated in (False, True):
            with self.subTest(simulated_windows_newlines=simulated):
                mode = patch.object(Path, "write_text", windows_text_write) if simulated else nullcontext()
                with mode, redirect_stdout(io.StringIO()):
                    operation()

    def test_foundations_lab_counts(self):
        def check():
            result = demo()
            self.assertEqual((result["characters"], result["bytes"]), (9, 13))
            self.assertEqual(result["lines"], ["hello", "学习"])
        self.run_both_modes(check)

    def test_f02_actual_example_bytes(self):
        def check():
            ns = {}
            exec(example(F02, "F02-lf-counts"), ns)
            self.assertEqual(ns["raw"], "hello\n学习\n".encode("utf-8"))
            self.assertEqual(len(ns["text"]), 9)
        self.run_both_modes(check)

    def test_a08_actual_read_example(self):
        self.run_both_modes(lambda: exec(example(A08, "A08-lf-read"), {}))

    def test_a08_actual_replace_example(self):
        def check():
            ns = {"Path": Path, "tempfile": tempfile, "Workspace": Workspace,
                  "ContractError": ContractError}
            definition(A08, "expect_error", ns)
            exec(example(A08, "A08-lf-replace"), ns)
        self.run_both_modes(check)

    def test_a14_actual_hash_example(self):
        def check():
            ns = {"Path": Path, "tempfile": tempfile}
            exec(example(A14, "A14-tax-source"), ns)
            exec(example(A14, "A14-lf-hash"), ns)
        self.run_both_modes(check)

    def test_coding_project_completes(self):
        def check():
            with tempfile.TemporaryDirectory() as directory:
                result = coding_project(directory)
                self.assertTrue(result["success"], result)
                self.assertEqual((result["before_exit"], result["after_exit"]), (1, 0))
                self.assertEqual((result["tool_calls"], result["model_calls"]), (3, 4))
                self.assertNotIn(b"\r\n", (Path(directory) / "price.py").read_bytes())
        self.run_both_modes(check)


class HostCatalogTests(unittest.TestCase):
    def setUp(self):
        self.ns = {"re": re}
        self.catalog_type = definition(A15, "HostCatalog", self.ns)
        self.catalog = self.catalog_type()
        self.search = Tool("search", "Search fixture", object_schema({}), lambda args: [])

    def test_same_tool_name_from_different_servers(self):
        for server in ("alpha", "beta"):
            name = self.catalog.add(server, "v1", self.search, {"learner"})
            self.assertEqual(name, server + "__search")
            self.assertEqual(self.catalog.resolve(name, "learner")[:3], (server, "v1", self.search))

    def test_delimiter_components_rejected_without_mutation(self):
        self.catalog.add("known", "v1", self.search, {"learner"})
        before = dict(self.catalog.entries)
        bad = Tool("beta__search", "Invalid name", object_schema({}), lambda args: [])
        for server, tool in (("alpha__beta", self.search), ("alpha", bad)):
            with self.subTest(server=server, tool=tool.name):
                with self.assertRaisesRegex(ValueError, "^invalid catalog name$"):
                    self.catalog.add(server, "v1", tool, {"learner"})
                self.assertEqual(self.catalog.entries, before)

    def test_duplicate_does_not_replace_original_version(self):
        name = self.catalog.add("alpha", "v1", self.search, {"learner"})
        with self.assertRaisesRegex(ValueError, "duplicate catalog entry"):
            self.catalog.add("alpha", "v2", self.search, {"other"})
        self.assertEqual(self.catalog.resolve(name, "learner")[1], "v1")
        with self.assertRaises(PermissionError):
            self.catalog.resolve(name, "other")

    def test_valid_underscore_and_length_boundaries(self):
        names = []
        for server, tool_name in (("a_", "b"), ("a", "b_"), ("a" * 32, "b" * 32)):
            tool = Tool(tool_name, "Fixture", object_schema({}), lambda args: [])
            name = self.catalog.add(server, "v1", tool, {"learner"})
            self.assertEqual(self.catalog.resolve(name, "learner")[0], server)
            names.append(name)
        self.assertEqual(len(set(names)), 3)
        for server in ("_alpha", "a" * 33, "Alpha", "", None):
            with self.assertRaises(ValueError):
                self.catalog.add(server, "v1", self.search, {"learner"})

    def test_unknown_and_unauthorized_names_are_unavailable(self):
        name = self.catalog.add("alpha", "v1", self.search, {"learner"})
        for candidate, subject in ((name, "stranger"), ("missing__search", "learner")):
            with self.assertRaises(PermissionError):
                self.catalog.resolve(candidate, subject)

    def test_actual_exercise_answer(self):
        self.ns.update(Tool=Tool, object_schema=object_schema, add_tool=arithmetic_tool())
        definition(A15, "expect_exception", self.ns)
        exec(example(A15, "A15-catalog-names"), self.ns)


def partitions(text):
    if not text:
        yield [""]
        return
    for mask in range(1 << (len(text) - 1)):
        cuts = [0] + [i for i in range(1, len(text)) if mask & (1 << (i - 1))] + [len(text)]
        yield [text[a:b] for a, b in zip(cuts, cuts[1:])]


class StopFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.filter_type = definition(S07, "StopFilter", {})

    def check_stream(self, text, stops, parts):
        hits = [text.find(stop) for stop in stops if stop in text]
        expected = text[:min(hits)] if hits else text
        stream = self.filter_type(stops)
        visible = ""
        for part in parts:
            visible += stream.feed(part)
            self.assertTrue(expected.startswith(visible), (text, stops, parts, visible, expected))
            self.assertLessEqual(len(stream.tail), max(map(len, stops)) - 1)
        visible += stream.finish()
        self.assertEqual((visible, stream.stopped), (expected, bool(hits)), (text, stops, parts))
        self.assertEqual(stream.finish(), "")
        if stream.stopped:
            self.assertEqual(stream.feed("ignored"), "")

    def test_overlap_waits_for_earlier_match_and_resolves_at_eof(self):
        stream = self.filter_type(["abcd", "bc"])
        self.assertEqual(stream.feed("abc"), "")
        self.assertFalse(stream.stopped)
        self.assertEqual(stream.feed("d"), "")
        self.assertTrue(stream.stopped)
        stream = self.filter_type(["abcd", "bc"])
        self.assertEqual(stream.feed("abc"), "")
        self.assertEqual(stream.finish(), "a")
        self.assertTrue(stream.stopped)

    def test_exhaustive_binary_texts_and_stop_pairs(self):
        pool = ["".join(chars) for n in range(1, 4) for chars in itertools.product("ab", repeat=n)]
        stop_sets = [(stop,) for stop in pool] + list(itertools.combinations(pool, 2))
        for length in range(7):
            for letters in itertools.product("ab", repeat=length):
                text = "".join(letters)
                for parts in partitions(text):
                    for stops in stop_sets:
                        self.check_stream(text, stops, parts)

    def test_unicode_empty_chunks_and_overlapping_stop_sets(self):
        cases = [("abcd", ["abcd", "bc"]), ("abc", ["abcd", "bc"]),
                 ("xabcdy", ["abcd", "bc"]), ("xabc", ["abcd", "bc"]),
                 ("学习结束以后", ["学习结束", "习结"]), ("xa", ["a", "abcd"]),
                 ("ababaa", ["aba", "bab", "aa"]), ("", ["!"])]
        for text, stops in cases:
            for parts in partitions(text):
                self.check_stream(text, stops, [""] + [piece for part in parts for piece in (part, "")])

    def test_actual_chapter_examples_and_exercise(self):
        ns = {}
        block = next(code for code in python_blocks(S07) if "class StopFilter:" in code)
        with redirect_stdout(io.StringIO()):
            exec(compile(block, S07, "exec"), ns)
            definition(S07, "offline_stop", ns)
            exec(example(S07, "S07-overlapping-stops"), ns)


class PositionOriginTests(unittest.TestCase):
    def test_m07_actual_cpu_chapter(self):
        # This chapter's Python blocks are all fixed offline NumPy examples.
        ns = {}
        with redirect_stdout(io.StringIO()):
            for code in python_blocks(M07):
                exec(compile(code, M07, "exec"), ns)
        self.assertIn("full_position_output", ns)
        self.assertIn("step_position_output", ns)


if __name__ == "__main__":
    unittest.main()
