#!/usr/bin/env python3
"""Focused tests for Stage-IV-era temporal, spatial, and MERRA-2 logic."""

import datetime as dt
import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

import my_functions_GPROF_V7_V8_MRMS_RAQI_ERA5_MERRA2_AutoSnow_StageIV_20260824 as mf
from collocate_GPROF_V7_V8_MRMS_RAQI_ERA5_MERRA2_AutoSnow_StageIV_20260824 import (
    inclusive_date_range,
)


class IntervalTests(unittest.TestCase):
    def test_boundaries_use_start_open_end_closed(self):
        expected = {
            dt.datetime(2021, 1, 1, 0, 0): dt.datetime(2021, 1, 1, 0, 0),
            dt.datetime(2021, 1, 1, 0, 25): dt.datetime(2021, 1, 1, 1, 0),
            dt.datetime(2021, 1, 1, 0, 59, 59): dt.datetime(2021, 1, 1, 1, 0),
            dt.datetime(2021, 1, 1, 1, 0): dt.datetime(2021, 1, 1, 1, 0),
        }
        for scan, end in expected.items():
            self.assertEqual(mf.accumulation_end_hour(scan), end)
            lower, upper = mf.accumulation_bounds(end)
            self.assertTrue(mf.interval_contains(scan, lower, upper))

    def test_exact_boundary_never_double_assigned(self):
        scan = dt.datetime(2021, 1, 1, 1)
        self.assertTrue(mf.interval_contains(scan, dt.datetime(2021, 1, 1, 0), scan))
        self.assertFalse(mf.interval_contains(scan, scan, dt.datetime(2021, 1, 1, 2)))

    def test_invalid_date_order(self):
        with self.assertRaisesRegex(ValueError, "on or before"):
            inclusive_date_range(dt.date(2021, 1, 2), dt.date(2021, 1, 1))


class ReaderTests(unittest.TestCase):
    def _write_stageiv(self, root, status=3.0):
        path = Path(root) / "2021_stage4_hourly.nc"
        time = np.array(["2021-01-01T01:00:00"], dtype="datetime64[s]")
        bounds = np.array(
            [["2021-01-01T00:00:00", "2021-01-01T01:00:00"]],
            dtype="datetime64[s]",
        )
        ds = xr.Dataset(
            {
                "time_bnds": (("time", "bnds"), bounds),
                "p01m": (("time", "y", "x"), np.array([[[1.0, 3.0], [5.0, 7.0]]])),
                "p01m_status": (("time",), np.array([status], np.float32)),
            },
            coords={
                "time": time,
                "lat": (("y", "x"), [[0.10, 0.10], [0.20, 0.20]]),
                "lon": (("y", "x"), [[0.10, 0.20], [0.10, 0.20]]),
            },
        )
        ds.to_netcdf(path)
        return path

    def test_stageiv_reads_one_hour_and_explicit_bounds(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._write_stageiv(root)
            with mf.StageIVAnnualReader(path, np.array([0.125]), np.array([0.125])) as reader:
                grid, end, start, status, counts = reader.read_hour(dt.datetime(2021, 1, 1, 1))
            self.assertAlmostEqual(float(grid[0, 0]), 4.0)
            self.assertEqual(int(counts[0, 0]), 4)
            self.assertEqual(status, 3.0)
            self.assertEqual(end - start, 3600.0)

    def test_stageiv_rejects_unset_status(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._write_stageiv(root, status=-1.0)
            with mf.StageIVAnnualReader(path, np.array([0.125]), np.array([0.125])) as reader:
                with self.assertRaisesRegex(ValueError, "unavailable status"):
                    reader.read_hour(dt.datetime(2021, 1, 1, 1))

    def test_merra2_retains_t2mwet(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "MERRA2_400.tavg1_2d_slv_Nx.20210101.SUB.nc"
            ds = xr.Dataset(
                {
                    "T2M": (("time", "lat", "lon"), np.full((1, 2, 2), 280, np.float32)),
                    "T2MWET": (("time", "lat", "lon"), np.full((1, 2, 2), 278, np.float32)),
                },
                coords={"time": np.array(["2021-01-01T00:30"], dtype="datetime64[m]"), "lat": [0, 1], "lon": [0, 1]},
            )
            ds["T2M"].attrs["units"] = "K"
            ds["T2MWET"].attrs["units"] = "K"
            ds.to_netcdf(path)
            t2m, wet, *_ = mf.read_merra2_temperatures(root, dt.datetime(2021, 1, 1, 0, 30))
            np.testing.assert_array_equal(t2m, 280)
            np.testing.assert_array_equal(wet, 278)

    def test_phase_regime(self):
        phase = mf.phase_regime(
            np.array([274.0, 280.0, 274.0]), np.array([273.0, 276.0, 274.0])
        )
        np.testing.assert_array_equal(phase, [1, 2, 0])


if __name__ == "__main__":
    unittest.main()
