# GPROF V7–V8 GMI precipitation evaluation

Research code for evaluating changes from GPROF Version 7 to Version 8 GMI
precipitation retrievals over the contiguous United States. The project uses
footprint-aligned and gridded comparisons with independent and complementary
references, including MRMS/RAQI, Stage IV, ERA5, MERRA-2, IMERG, and AutoSnow.

## Repository status

This is a code-and-technical-documentation archive for an active research
project. The source files are preserved in their existing form rather than
refactored into a software package. Input data, collocation archives, generated
statistics, figures, operational logs, and manuscript files are excluded.

The populated rain-server folder
`Comparing_V7_V8_GPROF-GMI_data/Codes` is the authoritative source for this
repository. A similarly named server folder was inspected and found to be empty.

## Workflow organization

The archive includes scripts for:

- pairing GPROF V7 and V8 GMI footprints and diagnosing swath availability;
- collocation with MRMS/RAQI, Stage IV, ERA5, MERRA-2, IMERG, and AutoSnow;
- hourly aggregation and temporal-matching diagnostics;
- precipitation-rate, occurrence, phase, surface, and seasonal evaluation;
- full-archive and study-period batch execution;
- quality-control checks and output validation; and
- analysis figures and manuscript-oriented summary diagnostics.

Start with [`RUNBOOK_footprint_aligned_20260925.md`](RUNBOOK_footprint_aligned_20260925.md)
and [`GPROF_V7_V8_MRMS_RAQI_COLLOCATION_TECHNICAL_REPORT.md`](GPROF_V7_V8_MRMS_RAQI_COLLOCATION_TECHNICAL_REPORT.md).
Dated and suffixed scripts preserve distinct development stages and should not
be assumed to be interchangeable.

## Associated manuscript

Kumah, K. K., Zandi, O., and Behrangi, A. (2026). *Evaluating Changes from
GPROF Version 7 to Version 8 GMI Precipitation Retrievals over the Contiguous
United States.* Manuscript in internal review prior to submission.

This citation and status match the author's public research website as of
October 2026. The status should be retained until a submission or publication
record is available.

## Data and execution

See [`DATA.md`](DATA.md) for the expected data categories. Many scripts retain
absolute rain-server paths as part of the analysis record. Configure equivalent
authorized paths locally before running them.

The workflows use scientific, geospatial, and satellite-data Python packages.
Exact dependency versions were not consistently recorded; inspect the imports
in the selected workflow and its helper modules before constructing an environment.

## License

No software license has yet been assigned. Contact the author before reuse or
redistribution.

## Contact

Kwabena Kingsley Kumah — [GitHub profile](https://github.com/King2coding)
