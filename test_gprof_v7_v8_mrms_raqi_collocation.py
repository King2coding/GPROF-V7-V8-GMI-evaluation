#!/usr/bin/env python3
"""Fast tests for temporal edges, pairing, geometry, and restart primitives."""

import datetime as dt
import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

import my_functions_gprof_v7_v8_mrms_raqi_era5_merra2_autosnow as mf
from collocate_gprof_v7_v8_with_mrms_raqi_era5_merra2_autosnow import (
    _nearest_merra2_time,
    inclusive_date_range,
)


class TemporalEdgeTests(unittest.TestCase):
    def test_explicit_date_range_is_inclusive(self):
        selected = inclusive_date_range(dt.date(2021, 1, 1), dt.date(2024, 12, 31))
        self.assertIn(dt.date(2021, 1, 1), selected)
        self.assertIn(dt.date(2024, 12, 31), selected)
        self.assertEqual(len(selected), 1461)

    def test_orbit_fully_inside_one_day(self):
        scans = [dt.datetime(2021, 1, 1, 12, 10), dt.datetime(2021, 1, 1, 12, 50)]
        self.assertEqual(mf.required_valid_hours(scans), [dt.datetime(2021, 1, 1, 12), dt.datetime(2021, 1, 1, 13)])

    def test_crossing_23xx_to_00xx(self):
        scans = [dt.datetime(2021, 1, 1, 23, 40), dt.datetime(2021, 1, 2, 0, 20)]
        self.assertEqual(mf.required_valid_hours(scans), [dt.datetime(2021, 1, 2, 0)])

    def test_first_scan_near_midnight(self):
        self.assertEqual(mf.nearest_hour(dt.datetime(2021, 1, 1, 0, 1)), dt.datetime(2021, 1, 1, 0))

    def test_last_scan_near_24(self):
        self.assertEqual(mf.nearest_hour(dt.datetime(2021, 1, 1, 23, 59)), dt.datetime(2021, 1, 2, 0))

    def test_half_hour_tie_is_earlier_like_legacy_nearest(self):
        self.assertEqual(mf.nearest_hour(dt.datetime(2021, 1, 1, 12, 30)), dt.datetime(2021, 1, 1, 12))

    def test_merra2_half_hour_centers_and_tie(self):
        self.assertEqual(_nearest_merra2_time(dt.datetime(2021, 1, 1, 1, 0)), dt.datetime(2021, 1, 1, 0, 30))

    def test_missing_previous_day_ancillary(self):
        with tempfile.TemporaryDirectory() as root:
            hour = dt.datetime(2021, 1, 1, 0)
            self.assertIsNone(mf.mrms_file(root, hour))
            self.assertIsNone(mf.raqi_file(root, hour))

    def test_missing_next_day_ancillary(self):
        with tempfile.TemporaryDirectory() as root:
            hour = dt.datetime(2021, 1, 2, 0)
            self.assertIsNone(mf.mrms_file(root, hour))
            self.assertIsNone(mf.raqi_file(root, hour))


class PairAndGeometryTests(unittest.TestCase):
    def test_pair_key_ignores_algorithm_and_version(self):
        v7 = mf.parse_granule_filename("2A-CLIM.GPM.GMI.GPROF2021v1.20210101-S010000-E020000.000001.V07A.HDF5")
        v8 = mf.parse_granule_filename("2A-CLIM.GPM.GMI.GPROFNNv1.20210101-S010000-E020000.000001.V08A.nc")
        self.assertEqual(v7["pair_key"], v8["pair_key"])

    def test_geometry_pass(self):
        base = {"lat": np.zeros((2, 3)), "lon": np.zeros((2, 3)), "scan_seconds": np.array([1., 2.])}
        self.assertTrue(mf.geometry_qc(base, {k: v.copy() for k, v in base.items()})["passed"])

    def test_geometry_fail_shape(self):
        a = {"lat": np.zeros((2, 3)), "lon": np.zeros((2, 3)), "scan_seconds": np.array([1., 2.])}
        b = {"lat": np.zeros((1, 3)), "lon": np.zeros((1, 3)), "scan_seconds": np.array([1.])}
        self.assertFalse(mf.geometry_qc(a, b)["passed"])

    def test_stale_partial_can_be_removed(self):
        with tempfile.TemporaryDirectory() as root:
            final = Path(root) / "x.nc"; partial = Path(str(final) + ".partial")
            partial.write_text("stale"); partial.unlink()
            self.assertFalse(partial.exists()); self.assertFalse(final.exists())

    def test_signed_time_difference_minutes(self):
        selected = np.array([3600.0, 7200.0])
        scans = np.array([2700.0, 8100.0])
        np.testing.assert_allclose(mf.time_difference_minutes(selected, scans), [15.0, -15.0])

    def test_quality_flags_are_explicit_and_consistent(self):
        mrms = np.array([[1.0, np.nan], [0.0, 2.0]], dtype=np.float32)
        raqi = np.array([[0.5, 0.2], [np.nan, 1.0]], dtype=np.float32)
        snow = np.array([[0, 255], [2, 3]], dtype=np.uint8)
        lat = np.array([[30, 30], [60, 30]], dtype=float)
        lon = np.array([[-100, -100], [-100, -50]], dtype=float)
        domain, mv, rq, sn, ref = mf.quality_flags(
            mrms, raqi, snow, lat, lon, (20, 55, 230, 300)
        )
        np.testing.assert_array_equal(domain, [[1, 1], [0, 0]])
        np.testing.assert_array_equal(mv, [[1, 0], [1, 1]])
        np.testing.assert_array_equal(rq, [[1, 1], [0, 1]])
        np.testing.assert_array_equal(sn, [[1, 0], [1, 1]])
        np.testing.assert_array_equal(ref, [[1, 0], [0, 1]])

    def test_era5_reader_returns_actual_selected_coordinate(self):
        with tempfile.TemporaryDirectory() as root:
            times = np.array(["2021-01-01T00:00", "2021-01-01T01:00"], dtype="datetime64[m]")
            ds = xr.Dataset(
                {"tp": (("valid_time", "latitude", "longitude"),
                        np.zeros((2, 2, 2), dtype=np.float32), {"units": "m"})},
                coords={"valid_time": times, "latitude": [1.0, 0.0], "longitude": [0.0, 1.0]},
            )
            ds.to_netcdf(Path(root) / "ERA5_tp_hourly_2021.nc")
            *_, selected = mf.read_era5_hour(root, dt.datetime(2021, 1, 1, 0, 40))
            expected = (dt.datetime(2021, 1, 1, 1) - mf.EPOCH).total_seconds()
            self.assertEqual(selected, expected)

    def test_merra2_reader_returns_actual_selected_coordinate(self):
        with tempfile.TemporaryDirectory() as root:
            times = np.array(["2021-01-01T00:30", "2021-01-01T01:30"], dtype="datetime64[m]")
            ds = xr.Dataset(
                {"T2M": (("time", "lat", "lon"),
                         np.full((2, 2, 2), 280.0, dtype=np.float32), {"units": "K"})},
                coords={"time": times, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
            )
            path = Path(root) / "MERRA2_400.tavg1_2d_slv_Nx.20210101.SUB.nc"
            ds.to_netcdf(path)
            *_, selected = mf.read_merra2_t2m(root, dt.datetime(2021, 1, 1, 1, 20))
            expected = (dt.datetime(2021, 1, 1, 1, 30) - mf.EPOCH).total_seconds()
            self.assertEqual(selected, expected)


if __name__ == "__main__":
    unittest.main()
