"""
tests/test_benchmark.py
========================
Unit tests for the Non-IID benchmark partition generator and related utilities.
Runs with standard unittest:
    python -m unittest tests/test_benchmark.py
"""

import os
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmark.partition_gen import (
    gini,
    overlap_gap,
    assert_no_leakage,
    recover_clock,
    mask_hash,
    quantity_skew,
    build_record,
)
from scripts.build_graph import time_label_of


class TestBenchmarkPartitionGen(unittest.TestCase):
    def test_gini_coefficient(self):
        # 1. Perfectly equal distribution -> Gini == 0.0
        equal_counts = [100, 100, 100, 100]
        self.assertAlmostEqual(gini(equal_counts), 0.0, places=5)

        # 2. Maximum inequality (one holds all, others 0) -> Gini close to 1.0
        extreme_counts = [1000, 0, 0, 0, 0]
        g = gini(extreme_counts)
        self.assertGreater(g, 0.75)

        # 3. Empty or all zero counts -> Gini == 0.0
        self.assertEqual(gini([]), 0.0)
        self.assertEqual(gini([0, 0, 0]), 0.0)

    def test_overlap_gap(self):
        meta = {"T_in": 12, "T_out": 3}
        self.assertEqual(overlap_gap(meta), 14)  # 12 + 3 - 1 = 14

        meta2 = {"T_in": 12, "T_out": 24}
        self.assertEqual(overlap_gap(meta2), 35)  # 12 + 24 - 1 = 35

    def test_assert_no_leakage(self):
        # Disjoint and distant train/test -> No exception
        train_idx = [0, 1, 2, 3, 4]
        test_idx = [30, 31, 32, 33]
        gap = 14
        assert_no_leakage(train_idx, test_idx, gap)

        # Overlapping samples -> ValueError
        with self.assertRaises(ValueError):
            assert_no_leakage([0, 1, 5], [5, 6, 7], gap=5)

        # Gap violation (min distance < gap) -> ValueError
        with self.assertRaises(ValueError):
            assert_no_leakage([0, 1, 2], [10, 11, 12], gap=10)  # distance is 10 - 2 = 8 < 10

    def test_recover_clock(self):
        S = 96 * 3  # 3 days
        t_in = 12
        hour, dow = recover_clock(S, t_in)

        self.assertEqual(len(hour), S)
        self.assertEqual(len(dow), S)
        self.assertTrue(np.all((hour >= 0) & (hour < 24)))
        self.assertTrue(np.all((dow >= 0) & (dow < 7)))

    def test_time_label_of(self):
        # 0 = night (0-5)
        for h in range(0, 6):
            self.assertEqual(time_label_of(h), 0, f"Hour {h} should be night (0)")

        # 1 = rush_morning (7-9)
        for h in range(7, 10):
            self.assertEqual(time_label_of(h), 1, f"Hour {h} should be rush_morning (1)")

        # 2 = rush_evening (16-19)
        for h in range(16, 20):
            self.assertEqual(time_label_of(h), 2, f"Hour {h} should be rush_evening (2)")

        # 3 = normal (6, 10-15, 20-23)
        normal_hours = [6] + list(range(10, 16)) + list(range(20, 24))
        for h in normal_hours:
            self.assertEqual(time_label_of(h), 3, f"Hour {h} should be normal (3)")

    def test_quantity_skew(self):
        S = 637
        N = 17
        alpha = 0.5
        seed = 42

        mask, stats = quantity_skew(S, N, alpha=alpha, seed=seed)

        self.assertEqual(mask.shape, (S, N))
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(stats["alpha"], alpha)
        self.assertEqual(stats["seed"], seed)
        self.assertEqual(len(stats["n_per_node"]), N)
        self.assertGreaterEqual(stats["gini"], 0.0)
        self.assertLessEqual(stats["gini"], 1.0)

        # Reproducibility test
        mask2, stats2 = quantity_skew(S, N, alpha=alpha, seed=seed)
        self.assertTrue(np.array_equal(mask, mask2))
        self.assertEqual(stats["gini"], stats2["gini"])


if __name__ == "__main__":
    unittest.main()
