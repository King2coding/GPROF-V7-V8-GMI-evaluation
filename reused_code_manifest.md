# Reused Code Manifest — Preliminary GMI GPROF V7/V8 Analysis

This manifest records the provenance and scientific review of functions copied
or refactored into `preliminary_gmi_gprof_v7_v8_analysis.py`. No source module
is imported because several older project scripts execute analysis at import
time. The original files are not modified.

## Continuous verification metrics

- Original source path:
  `/home/kkumah/Projects/Satellite_eval_over_Oceans/codes/IEPPO_utils.py`
- Original section: `calculate_metrics` (near line 290), together with its
  `relative_bias`, RMSE, Pearson-correlation, and MAE helpers.
- New component: `PairStats`.
- Purpose: exact incremental continuous metrics for product versus MRMS.
- Modifications: removed early rounding; added sample count, reference and
  product means, mean bias, population standard deviations, and centered RMSE;
  replaced whole-array evaluation with mergeable sums so native orbit arrays
  are processed one file at a time.
- Formulae:
  `bias = mean(product-reference)`;
  `relative_bias = 100*sum(product-reference)/sum(reference)`;
  `MAE = mean(abs(product-reference))`;
  `RMSE = sqrt(mean((product-reference)^2))`;
  Pearson correlation is covariance divided by the product of population
  standard deviations; `centered_RMSE = sqrt(mean(((product-mean(product)) -
  (reference-mean(reference)))^2))`.

## Contingency table and categorical metrics

- Original source path:
  `/home/kkumah/Projects/Satellite_eval_over_Oceans/codes/IEPPO_utils.py`
- Original section: `categorical_stats` (near line 1417).
- New component: `ContingencyStats`.
- Purpose: categorical precipitation verification at 0.1.
- Modifications: retained the `>= threshold` event definition and contingency
  counts; added success ratio and equitable threat score; removed rounding;
  made counts incrementally mergeable.
- Formulae: `POD=H/(H+M)`; `FAR=F/(H+F)`;
  `success_ratio=H/(H+F)=1-FAR`; `CSI=H/(H+M+F)`;
  `frequency_bias=(H+F)/(H+M)`;
  `H_random=(H+M)(H+F)/N` and
  `ETS=(H-H_random)/(H+M+F-H_random)`.

## Roebber performance-diagram background

- Original source path:
  `/home/kkumah/Projects/Satellite_eval_over_Oceans/codes/IEPPO_utils.py`
- Original section: `draw_perf_background` (near line 11536).
- New component: `draw_performance_background`.
- Purpose: CSI contours and frequency-bias guides in success-ratio/POD space.
- Modifications: simplified typography and palette for the preliminary report;
  retained the scientific geometry.
- Formulae: `CSI = 1/(1/SR + 1/POD - 1)` and
  `frequency_bias=POD/SR`, so guide lines satisfy `POD=bias*SR`.

## Precipitation occurrence PDFs

- Original source path:
  `/home/kkumah/Projects/Satellite_eval_over_Oceans/codes/IEPPO_utils.py`
- Original section: `compute_pdf_elements` (near line 1562).
- New component: `OccurrenceDistribution`.
- Purpose: common-bin occurrence-frequency PDFs.
- Modifications: vectorized with `numpy.histogram`; separated exact zero
  precipitation from positive bins; added an overflow bin; accumulated counts
  incrementally; did not reuse the precipitation-volume weighting because the
  requested product is an occurrence PDF.
- Formula: each bin percentage is `100 * bin_count / selected_sample_count`.

## Independently implemented components

No suitable reusable implementation was found in the inspected `codes` or
`Codes` directories for the Taylor diagram, the incremental log1p density
histograms, temperature-bin aggregation, or AutoSnow-class aggregation.
These components were implemented directly in the new module and are covered
by its validation checks and focused unit tests.

- Taylor diagram: polar angle `theta=arccos(correlation)`, radius
  `sigma_product/sigma_reference`, and normalized centered RMSE contours
  `sqrt(1+r^2-2*r*cos(theta))`.
- Scatter density: every selected footprint contributes to a two-dimensional
  histogram in `(log1p(MRMS), log1p(product))` space; no point subsampling is
  used.
