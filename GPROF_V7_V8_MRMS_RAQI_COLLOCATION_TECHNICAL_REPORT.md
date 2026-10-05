# GMI GPROF V7/V8 MRMS–RAQI collocation technical report

## Phase-1 scope

The production framework is GMI-only. Each NetCDF represents one paired GMI
GPROF V7/V8 orbit on the complete native V7 `scan x pixel` geometry. MHS,
SSMIS, and cross-sensor L2 matching are explicitly deferred to Phase 2. The
configuration structure remains extensible, but the command-line and shell
runner defaults are now `--sensor GMI`.

The output contains:

- GPROF V7 and V8 `S1/surfacePrecipitation`
- MRMS `MultiSensor_QPE_01H_Pass2`
- reconstructed hourly `RAQI`
- ERA5 `tp`
- MERRA-2 `T2M`
- GMASI/AutoSnow `autosnow_class`
- selected ancillary times, signed time differences, and explicit quality flags

## Implementation files

- `my_functions_gprof_v7_v8_mrms_raqi_era5_merra2_autosnow.py`
- `collocate_gprof_v7_v8_with_mrms_raqi_era5_merra2_autosnow.py`
- `run_full_collocation_gprof_v7_v8_mrms_raqi_era5_merra2_autosnow.sh`
- `validate_gmi_v7_v8_mrms_raqi_outputs.py`
- `test_gprof_v7_v8_mrms_raqi_collocation.py`

## Pairing and geometry

V7 and V8 filenames are paired using the legacy key
`YYYYMMDD-SHHMMSS-EHHMMSS.orbit`. Algorithm and version tokens are deliberately
excluded. V7 latitude, longitude, scan timestamps, scan count, and footprint
count are authoritative.

Geometry QC checks shape, scan time within one second, latitude difference,
and wrapped-longitude difference. Normal tolerance is 0.005 degrees. The
existing narrowly defined minor edge exception is retained; true mismatches
are logged and skipped.

## Temporal conventions

All selection uses each native V7 scan's decoded UTC timestamp:

- MRMS: nearest hourly valid time; exact half-hour ties select earlier.
- RAQI: exactly the same selected valid hour as MRMS.
- ERA5: nearest hourly valid time; exact half-hour ties select earlier.
- MERRA-2: nearest available hourly field (source fields centered at `:30`);
  exact ties select earlier.
- AutoSnow: daily map for the actual UTC scan date.

The output stores `MRMS_valid_time`, `RAQI_time`, `ERA5_time`, and
`MERRA2_time`. It also stores signed ancillary-minus-GMI offsets in minutes:

- `MRMS_time_difference_minutes`
- `RAQI_time_difference_minutes`
- `ERA5_time_difference_minutes`
- `MERRA2_time_difference_minutes`

This preserves tolerance information for later sensitivity analysis without
rerunning collocation.

## Spatial harmonization

The actual latitude and longitude arrays in the ERA5 yearly file define the
common grid. All harmonization is in memory:

- MRMS, RAQI, and MERRA-2 use `rasterio.warp.reproject` with
  `Resampling.average` and source missing values excluded.
- AutoSnow uses `Resampling.mode` with 255 as missing.
- ERA5 remains native.
- The nearest ERA5-grid center is sampled back to every native V7 footprint
  center.

No resampled meteorological files or uncompressed GRIB files are written.
MRMS `.grib2.gz` is decompressed and decoded in memory with ecCodes.

AutoSnow classes are 0 clear water, 1 snow-free land, 2 snow-covered land,
and 3 ice-covered water. Invalid or missing values are 255.

## Edge handling and restart safety

Ancillary dates are derived independently for each scan. Previous, current,
and next-day inputs are opened only when selected timestamps require them.
Scans are not clipped, duplicated, or assigned solely from the filename date.

The completed output path is checked before opening either GPROF file or any
large ancillary field. New files are written as `.partial` and atomically
renamed. Stale partial outputs are removed. `--overwrite` explicitly rebuilds
a completed file.

## Quality flags

Five `uint8` flags remove the need to infer analysis validity only from NaNs:

- `mrms_domain_flag`: native footprint center is within the native MRMS
  center-coordinate bounding box. This is geometry metadata only and is not a
  collocated-data validity test.
- `mrms_valid_flag`: `isfinite(MRMS_Pass2)` after ERA5 harmonization and
  footprint sampling. This is the authoritative MRMS validity flag.
- `raqi_valid_flag`: collocated RAQI is finite.
- `autosnow_valid_flag`: AutoSnow is a valid class rather than 255.
- `valid_reference_flag`: both MRMS and RAQI are finite.

Each flag uses 0=false/invalid and 1=true/valid. The strict validator recomputes
all validity definitions and the native bounding-box test and requires exact
equality. A finite zero MRMS accumulation remains valid.

## Metadata

Every data variable and coordinate includes a long name, units, source dataset,
source variable, spatial method, temporal method, and missing-value handling.
Global attributes record source files, pairing, geometry QC, common ERA5 grid,
time conventions, spatial harmonization, edge behavior, software version, and
UTC creation time.

## Representative validation

Three GMI orbits from 2021-01-01 were atomically regenerated and passed the
strict validator:

| Orbit | Dimensions | Geometry | Scan range | Midnight |
|---|---:|---|---|---|
| 038890 | 2963 x 221 | PASS_NORMAL | 13:52:09–15:24:42 | no |
| 038895 | 2962 x 221 | PASS_NORMAL | 21:35:01–23:07:33 | no |
| 038896 | 2963 x 221 | PASS_NORMAL | Jan 1 23:07:35–Jan 2 00:40:09 | yes |

For all three, MRMS, RAQI, ERA5, and MERRA-2 signed time differences span
approximately -29.98 to +29.98 minutes, consistent with nearest-hour matching.
Every stored ancillary time was independently matched to an actual source
coordinate or MRMS source validity stamp. Times are assigned only after the
field is successfully loaded and sampled; failed fields retain missing data,
selected time, and time-difference diagnostics. MRMS and RAQI selected times
are identical where both load. RAQI remains within 0–1, MERRA-2
T2M is within 225.59–308.03 K, all precipitation is nonnegative, and AutoSnow
contains only classes 0–3.

MRMS and RAQI are missing over roughly 88–92% of the full orbit because those
references cover CONUS while GMI orbits are global. This is expected and is
made explicit by domain and validity flags.

The validation summary is written to:

`/scratch/kkumah/GPROF_V7_V8_MRMS_RAQI_ERA5_MERRA2_AutoSnow_matchups/qc/gmi_validation_summary.json`

Three quick looks are generated per orbit under `qc/figures`: a scan–pixel
field coverage figure, a signed time-offset figure, and an MRMS geographic
boundary figure. GPROF color scales are visibly documented as capped at the
joint 99.5th percentile for pattern readability.

Sixteen focused unit/edge tests pass, including midnight boundaries, missing
previous/next-day paths, exact half-hour behavior, pairing, geometry, signed
offset calculations, quality-flag definitions, actual xarray selected-time
coordinates, and stale-partial handling.

## Additional pre-production MRMS boundary QC

The regenerated prototypes produced the requested counts:

| Orbit | `domain=0` and finite MRMS | `domain=1` and nonfinite MRMS | Outside finite zero | Outside finite positive |
|---|---:|---:|---:|---:|
| 038890 | 356 | 242 | 356 | 0 |
| 038895 | 365 | 242 | 295 | 70 |
| 038896 | 265 | 248 | 110 | 155 |

The native MRMS center bounds are 20.005001–54.995°N and
230.005–299.994998°E; the native outer cell edges are essentially
20–55°N and 230–300°E. After average resampling, finite ERA5 cell centers span
20.25–55°N and 230–300°E. Nearest-center footprint sampling therefore extends
the northern and western finite-data reach by approximately half an ERA5 cell,
to 55.125°N and 229.875°E, while the southern finite boundary contracts to
approximately 20.125°N. All `domain=1`/nonfinite cases occur in this narrow
southern boundary ring; none occur in the interior.

Valid zero precipitation explains part of the visible extra swath, but cannot
explain it alone: 225 outside-domain finite samples across the three orbits are
positive. The primary geometric cause is ERA5-grid boundary alignment followed
by nearest-center sampling. This is expected support expansion from the stated
harmonization method, not an MRMS GRIB orientation defect.

The real GRIB scan keys are `iScansNegatively=0`, `jScansPositively=0`,
`jPointsAreConsecutive=0`, and `alternativeRowScanning=0`. An independent
ecCodes latitude/longitude-array decode reproduces the known corners
(54.995°N, 230.005°E), (54.995°N, 299.994998°E),
(20.005001°N, 230.005°E), and (20.005001°N, 299.994998°E), with -0.01° row
steps and +0.01° column steps. Thus `reshape(Nj, Ni)`, north-to-south latitude,
and west-to-east longitude are correct.

## Remaining limitations

- MRMS/RAQI have regional rather than global spatial coverage.
- ERA5 `tp` and MRMS QPE are one-hour accumulations in millimeters; they should
  not be treated as instantaneous precipitation rates without an explicit
  conversion/interpretation step.
- Spatial sampling uses the ERA5 cell containing the nearest grid center, not a
  full GMI antenna-pattern convolution.
- The existing GMI V7/V8 retrieval differences and differing missing fractions
  are preserved; no attempt is made to impute either retrieval.
- Phase-2 L2 cross-sensor matching has not been implemented.

## Commands

Discovery only:

```bash
/home/kkumah/.conda/envs/cml_env/bin/python \
  collocate_gprof_v7_v8_with_mrms_raqi_era5_merra2_autosnow.py \
  --sensor GMI --workers 2 --discovery-only
```

Validation:

```bash
/home/kkumah/.conda/envs/cml_env/bin/python \
  validate_gmi_v7_v8_mrms_raqi_outputs.py
```

Production after scientific review:

```bash
WORKERS=2 ./run_full_collocation_gprof_v7_v8_mrms_raqi_era5_merra2_autosnow.sh
```

The full production archive has not been launched.
