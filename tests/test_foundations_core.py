import asyncio
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from contextlib import nullcontext
from unittest.mock import patch

from labs.foundations import paths, permissions, process_runner, http_message, json_contract
from labs.foundations import python_values, functions, objects, iterators, file_io, sqlite_lab
from labs.foundations import debugging, cli, concurrency, git_model, container_model
from labs.foundations import text_pipeline, tensors, numerics

class FoundationsCoreTests(unittest.TestCase):
    def test_paths_counts_unicode_bytes(self):
        result = paths.demo()
        self.assertEqual(result['characters'], 9)
        self.assertEqual(result['bytes'], 13)
        self.assertEqual(result['lines'], ['hello', '学习'])

    def test_paths_reject_escape(self):
        with TemporaryDirectory() as d:
            root = Path(d).resolve()
            with self.assertRaises(ValueError): paths.contained_path(root, '../escape')
            with self.assertRaises(ValueError): paths.contained_path(root, str(root.resolve()))
            self.assertEqual(paths.contained_path(root, 'a/../b'), root / 'b')

    def test_paths_demo_normalizes_temporary_root(self):
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            relative = Path(directory).relative_to(Path.cwd())
            with patch.object(paths, 'TemporaryDirectory', return_value=nullcontext(str(relative))):
                result = paths.demo()
            self.assertEqual(Path(result['relative']), Path('notes') / 'lesson.txt')
            self.assertEqual(result['bytes'], 13)

    def test_permissions(self):
        self.assertEqual(permissions.symbolic(0o640), 'rw-r-----')
        self.assertEqual(permissions.apply_umask(0o666, 0o022), 0o644)
        with self.assertRaises(ValueError): permissions.symbolic(0o1000)

    def test_child_arguments_are_literal(self):
        text = '; $(not-an-executable) *'
        self.assertEqual(process_runner.run_child(text), dict(stdout=text, stderr='diagnostic', code=0))

    def test_http_split(self):
        value = http_message.describe('https://example.com/a?x=1&x=2#top')
        self.assertEqual(value['port'], 443)
        self.assertEqual(value['query'], {'x':['1','2']})
        self.assertFalse(value['fragment_sent_to_server'])
        with self.assertRaises(ValueError): http_message.describe('file:///etc/passwd')

    def test_json_contract_valid(self):
        self.assertEqual(json_contract.parse_request('{"request_id":"r","tokens":0}').tokens, 0)

    def test_json_contract_rejects_boolean_and_shape(self):
        for text in ['{"request_id":"r","tokens":true}', '[]', '{"request_id":" ","tokens":1}']:
            with self.assertRaises(ValueError): json_contract.parse_request(text)

    def test_aliases(self):
        original, shallow, deep = python_values.aliases()
        self.assertEqual(original[0], [1,2,9])
        self.assertIs(original[0], shallow[0])
        self.assertEqual(deep[0], [1,2])

    def test_batches_preserve_every_item(self):
        for count in range(20):
            for size in range(1,7):
                batches = functions.batches(range(count), size)
                self.assertEqual([x for b in batches for x in b], list(range(count)))
                self.assertTrue(all(0<len(b)<=size for b in batches))
        with self.assertRaises(ValueError): functions.batches([1],0)

    def test_cost(self):
        self.assertAlmostEqual(functions.token_cost(1000,200,input_rate=2,output_rate=8), .0036)

    def test_budget_failure_preserves_state(self):
        budget = objects.Budget(10)
        budget.consume(7)
        with self.assertRaises(ValueError): budget.consume(4)
        self.assertEqual(budget.remaining,3)
        with self.assertRaises(ValueError): budget.consume(True)

    def test_iterator_lazy(self):
        visits = []
        def source():
            for i in range(10):
                visits.append(i)
                yield i
        stream = iterators.stream_batches(source(), 3)
        self.assertEqual(visits, [])
        self.assertEqual(next(stream), [0,1,2])
        self.assertEqual(visits,[0,1,2])
        stream.close()

    def test_atomic_json_failure_preserves_previous(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'a.json'
            file_io.atomic_json(p, {'value':1})
            with self.assertRaises(ValueError): file_io.atomic_json(p, {'value':float('nan')})
            self.assertEqual(json.loads(p.read_text()), {'value':1})
            self.assertEqual([x.name for x in Path(d).iterdir()], ['a.json'])

    def test_sqlite_rollback(self):
        self.assertEqual(sqlite_lab.transaction_demo(), [('tokens',10)])

    def test_mean_edges(self):
        self.assertEqual(debugging.mean([5]),5)
        with self.assertRaises(ValueError): debugging.mean([])

    def test_cli(self):
        output=io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(cli.main(['--count','5','--batch-size','2']),0)
        self.assertEqual(json.loads(output.getvalue()), [[0,1],[2,3],[4]])

    def test_bounded_async(self):
        result, meter=asyncio.run(concurrency.squares(range(9),3))
        self.assertEqual(result,[i*i for i in range(9)])
        self.assertLessEqual(meter.peak,3)
        self.assertEqual(meter.active,0)

    def test_git_staging_snapshot(self):
        repo=git_model.Snapshot()
        repo.work['a']='v1'; repo.add('a'); repo.work['a']='v2'; repo.commit()
        self.assertEqual(repo.commits[-1][1]['a'],'v1')
        repo.index['a']='v3'
        self.assertEqual(repo.commits[-1][1]['a'],'v1')

    def test_container_shared_volume_separate_layer(self):
        image={'/x':'old'}; volume={}
        a=container_model.Container(image, volume=volume)
        a.write('/x','new'); a.write('/data/x','persist')
        b=container_model.Container(image, volume=volume)
        self.assertEqual(b.read('/x'),'old')
        self.assertEqual(b.read('/data/x'),'persist')

    def test_pipeline(self):
        self.assertEqual(text_pipeline.error_counts(text_pipeline.SAMPLE), {'invalid_json':1,'timeout':2})

    def test_matrix(self):
        import numpy as np
        x,w,y=tensors.demo()
        np.testing.assert_array_equal(y,[[12,23],[18,29]])
        self.assertTrue(np.shares_memory(x,x.T))

    def test_softmax_shift_invariant(self):
        import numpy as np
        np.testing.assert_allclose(numerics.softmax([1,2,3]),numerics.softmax([1001,1002,1003]))
        self.assertAlmostEqual(float(numerics.softmax([1000,1001]).sum()),1)
        self.assertAlmostEqual(numerics.finite_difference(lambda x:x*x,3),6,places=6)
        with self.assertRaises(ValueError): numerics.softmax([])
        with self.assertRaises(ValueError): numerics.softmax([float('inf')])

if __name__=='__main__': unittest.main()
