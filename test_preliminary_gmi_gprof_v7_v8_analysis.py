#!/usr/bin/env python3
"""Focused scientific-formula tests for the preliminary GMI analysis."""

import unittest

import numpy as np

import preliminary_gmi_gprof_v7_v8_analysis as analysis


class PairMetricTests(unittest.TestCase):
    def test_pair_stats_matches_direct_calculation_and_chunking(self):
        reference = np.array([0.0, 1.0, 2.0, 4.0])
        product = np.array([0.0, 2.0, 1.0, 5.0])
        stats = analysis.PairStats()
        stats.update(reference[:2], product[:2])
        stats.update(reference[2:], product[2:])
        result = stats.result()
        error = product - reference
        self.assertEqual(result["N"], 4)
        self.assertAlmostEqual(result["mean_bias"], np.mean(error))
        self.assertAlmostEqual(result["mae"], np.mean(np.abs(error)))
        self.assertAlmostEqual(result["rmse"], np.sqrt(np.mean(error ** 2)))
        self.assertAlmostEqual(result["pearson_r"], np.corrcoef(reference, product)[0, 1])
        centered = (product - product.mean()) - (reference - reference.mean())
        self.assertAlmostEqual(result["centered_rmse"], np.sqrt(np.mean(centered ** 2)))

    def test_relative_bias_zero_reference_is_missing(self):
        stats = analysis.PairStats()
        stats.update([0.0, 0.0], [1.0, 2.0])
        self.assertTrue(np.isnan(stats.result()["relative_bias_percent"]))


class CategoricalMetricTests(unittest.TestCase):
    def test_contingency_and_ets(self):
        reference = np.array([0.0, 0.1, 0.2, 0.0])
        product = np.array([0.0, 0.2, 0.0, 0.1])
        stats = analysis.ContingencyStats()
        stats.update(reference, product, threshold=0.1)
        result = stats.result()
        self.assertEqual((result["hits"], result["misses"], result["false_alarms"],
                          result["correct_negatives"]), (1, 1, 1, 1))
        self.assertAlmostEqual(result["POD"], 0.5)
        self.assertAlmostEqual(result["FAR"], 0.5)
        self.assertAlmostEqual(result["success_ratio"], 0.5)
        self.assertAlmostEqual(result["CSI"], 1 / 3)
        self.assertAlmostEqual(result["frequency_bias"], 1.0)
        self.assertAlmostEqual(result["ETS"], 0.0)


class DistributionAndMaskTests(unittest.TestCase):
    def test_zero_is_valid_and_distribution_conserves_count(self):
        distribution = analysis.OccurrenceDistribution()
        distribution.update(np.array([0.0, 0.005, 0.1, 200.0]))
        self.assertEqual(distribution.zero_count, 1)
        self.assertEqual(distribution.total_count, 4)
        self.assertEqual(int(distribution.positive_counts.sum()), 3)

    def test_four_product_mask_is_subset(self):
        arrays = {
            "surfacePrecipitation_V7": np.array([0.0, 1.0, 1.0]),
            "surfacePrecipitation_V8": np.array([0.0, 1.0, 1.0]),
            "MRMS_Pass2": np.array([0.0, 1.0, 1.0]),
            "RAQI": np.array([0.7, 0.7, 0.7]),
            "ERA5_precipitation": np.array([0.0, np.nan, 1.0]),
            "mrms_valid_flag": np.array([1, 1, 0]),
        }
        v7v8, four = analysis.base_masks(arrays)
        np.testing.assert_array_equal(v7v8, [True, True, False])
        np.testing.assert_array_equal(four, [True, False, False])


if __name__ == "__main__":
    unittest.main()
