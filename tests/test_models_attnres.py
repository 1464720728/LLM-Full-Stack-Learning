import unittest
from unittest.mock import patch
import numpy as np

from labs.models.block_attnres_decoder import TinyBlockAttnResDecoder
from labs.models.core import DecoderConfig, rms_norm


class BlockAttentionResidualTests(unittest.TestCase):
    def model(self, block=4):
        return TinyBlockAttnResDecoder(DecoderConfig(layers=3, dim=8, heads=2,
                                                     kv_heads=1, hidden=12), block)

    def test_source_counts_and_boundary_order(self):
        for block, expected in [(2, [1, 2, 2, 3, 3, 4, 4]),
                                (4, [1, 2, 2, 2, 2, 3, 3]),
                                (8, [1, 2, 2, 2, 2, 2, 2])]:
            _, audit = self.model(block).forward([1, 2], return_trace=True)
            events = audit["events"]
            self.assertEqual([len(e["sources"]) for e in events], expected)
            self.assertEqual(events[0]["source_labels"], ("embedding",))
            self.assertEqual(events[0]["sealed_after_mix"], "embedding")
            # At depth 4, the old partial participates BEFORE it is sealed.
            if block == 4:
                self.assertEqual(events[4]["source_labels"], ("embedding", "partial[0:4]"))
                self.assertEqual(events[4]["sealed_after_mix"], "block[0:4]")
                self.assertEqual(events[5]["source_labels"],
                                 ("embedding", "block[0:4]", "partial[4:5]"))

    def test_zero_queries_average_raw_sources(self):
        _, audit = self.model().forward([1, 2, 3], return_trace=True)
        for event in audit["events"]:
            np.testing.assert_allclose(event["mixed"], event["sources"].mean(0), atol=1e-13)
            np.testing.assert_allclose(event["weights"], 1 / len(event["sources"]), atol=1e-13)
            np.testing.assert_allclose(event["weights"].sum(0), 1)

    def test_full_cached_equivalence_with_content_dependent_queries(self):
        for block in (2, 4, 8):
            model = self.model(block)
            model.depth_queries[:] = np.random.default_rng(32).normal(size=model.depth_queries.shape)
            ids = [1, 1, 4, 2, 8]
            state, actual = None, []
            for token in ids:
                logits, state = model.step(token, state)
                actual.append(logits)
            np.testing.assert_allclose(actual, model.forward(ids), rtol=1e-11, atol=1e-11)

    def test_all_prefixes_are_independent_of_future_tokens(self):
        model = self.model()
        model.depth_queries[:] = np.random.default_rng(3).normal(size=model.depth_queries.shape)
        ids = [1, 2, 3, 4, 5]
        full = model.forward(ids)
        for prefix in range(1, len(ids)):
            changed = ids[:prefix] + [9] * (len(ids) - prefix)
            np.testing.assert_allclose(model.forward(changed)[:prefix], full[:prefix], atol=1e-12)
            np.testing.assert_allclose(model.forward(ids[:prefix]), full[:prefix], atol=1e-12)

    def test_snapshots_own_storage_and_remain_unchanged(self):
        model = self.model()
        _, audit = model.forward([1, 2], return_trace=True)
        sealed = audit["completed_sources"]
        initial, later = sealed[0], sealed[1]
        np.testing.assert_array_equal(initial, model.embedding[[1, 2]])
        self.assertFalse(np.shares_memory(initial, model.embedding))
        self.assertFalse(np.shares_memory(initial, later))
        self.assertFalse(np.shares_memory(later, audit["partial"]))
        np.testing.assert_array_equal(later, audit["events"][4]["sources"][-1])
        before = later.copy()
        model.embedding[:] += 10
        model.forward([1, 2, 3])
        np.testing.assert_array_equal(later, before)
        with self.assertRaises(ValueError):
            later[0, 0] = 999

    def test_all_sublayers_participate_in_logits(self):
        model = self.model()
        model.depth_queries[:] = 0.1
        original = model.forward([1, 2, 3])
        for block in model.blocks:
            for name in ("o", "down"):
                saved = block[name].copy()
                block[name][:] = 0
                difference = np.max(abs(model.forward([1, 2, 3]) - original))
                self.assertGreater(difference, 1e-6)
                block[name][:] = saved

    def test_zero_updates_preserve_embedding_direction_without_duplicates(self):
        model = self.model()
        for block in model.blocks:
            block["o"][:] = 0
            block["down"][:] = 0
        logits, audit = model.forward([1, 2], return_trace=True)
        final = audit["events"][-1]
        # Completed embedding + completed zero block + current zero partial.
        self.assertEqual(final["source_labels"], ("embedding", "block[0:4]", "partial[4:6]"))
        np.testing.assert_allclose(final["mixed"], model.embedding[[1, 2]] / 3)
        np.testing.assert_allclose(logits, rms_norm(model.embedding[[1, 2]] / 3) @ model.output)

    def test_known_nonzero_deltas_are_neither_duplicated_nor_lost(self):
        # An independent lifecycle oracle: attention adds 1,2,3,4,5;
        # MLP adds zero. At zero queries all final sources average equally.
        # Hence the numerator is embedding + 15 for EVERY block partition.
        for size in (2, 4, 6, 8):
            model = TinyBlockAttnResDecoder(DecoderConfig(layers=5), size)
            index_by_id = {id(block): i for i, block in enumerate(model.blocks)}

            def constant_attention(x, block, position, previous=None):
                return np.full_like(x, index_by_id[id(block)] + 1), None

            model._attn = constant_attention
            with patch("labs.models.block_attnres_decoder.swiglu",
                       side_effect=lambda x, *weights: np.zeros_like(x)):
                _, audit = model.forward([1, 2], return_trace=True)
            final = audit["events"][-1]
            count = 2 + (2 * model.config.layers - 1) // size
            np.testing.assert_allclose(final["mixed"],
                                       (model.embedding[[1, 2]] + 15) / count)

    def test_step_does_not_mutate_input_cache(self):
        model = self.model()
        _, cache = model.step(1)
        saved = [(k.copy(), v.copy()) for k, v in cache]
        _, new = model.step(2, cache)
        for old, old_copy, current in zip(cache, saved, new):
            for a, b, c in zip(old, old_copy, current):
                np.testing.assert_array_equal(a, b)
                self.assertFalse(np.shares_memory(a, c))

    def test_invalid_sizes_ids_queries_and_cache_are_rejected(self):
        for block in (0, 1, 3, -2, True, 2.0):
            with self.assertRaises(ValueError):
                self.model(block)
        model = self.model()
        for ids in ([], [1.5], [True], [-1], [999]):
            with self.assertRaises(ValueError):
                model.forward(ids)
        with self.assertRaises(ValueError):
            model.step(True)
        _, cache = model.step(1)
        with self.assertRaises(ValueError):
            model.step(2, cache[:1])
        with self.assertRaises(ValueError):
            model.step(2, [None] + cache[1:])
        model.depth_queries[0, 0] = np.nan
        with self.assertRaises(ValueError):
            model.forward([1])


if __name__ == "__main__":
    unittest.main()
