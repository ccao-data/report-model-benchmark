
This repository compares the time-to-train of two gradient-boosted
decision tree (GBDT) frameworks - LightGBM and XGBoost - across
different hardware and versions. The purpose of this comparison is to
help the CCAO make two decisions:

1.  Which GBDT framework to use for its 2024 automated valuation models
2.  Whether or not to purchase additional hardware (a GPU) in order to
    improve model training speed

Below are the results of our tests. ***Please note that the performance
statistics presented here are only for cross-model comparison and do not
reflect any real model results e.g. they are included only to show that
each framework and version generates similar results, given the same
data.***

| Server | Type                                                                                                                         | Device | Package Version | Approx. Peak Device Utilization | Wall Time (Full Run) | Wall Time (Prediction) | Wall Time (SHAP) | RMSE      | MAE      | MAPE   | R2    | COD    | PRD   | PRB    | MKI   |
|:-------|:-----------------------------------------------------------------------------------------------------------------------------|:-------|:----------------|--------------------------------:|---------------------:|-----------------------:|-----------------:|:----------|:---------|:-------|:------|:-------|:------|:-------|:------|
| CCAO   | LightGBM                                                                                                                     | CPU    | 3.3.5           |                            100% |               3m 55s |                  1m 5s |        2h 1m 38s | \$130,989 | \$74,124 | 26.97% | 0.883 | 27.634 | 1.140 | −0.225 | 0.851 |
| CCAO   | LightGBM                                                                                                                     | CPU    | 4.1.0           |                            100% |                4m 8s |                  1m 4s |        2h 1m 48s | \$130,989 | \$74,124 | 26.97% | 0.883 | 27.634 | 1.140 | −0.225 | 0.851 |
| CCAO   | XGBoost<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span>  | CPU    | 2.0.0.1         |                            100% |               2m 41s |                    12s |           2m 14s | \$126,389 | \$74,129 | 26.76% | 0.885 | 27.458 | 1.130 | −0.212 | 0.866 |
| NVIDIA | LightGBM                                                                                                                     | CPU    | 3.3.5           |                            100% |               4m 12s |                     9s |           19m 5s | \$130,989 | \$74,124 | 26.97% | 0.883 | 27.634 | 1.140 | −0.225 | 0.851 |
| NVIDIA | LightGBM                                                                                                                     | CPU    | 4.1.0.99        |                            100% |                4m 7s |                     9s |          18m 52s | \$130,989 | \$74,124 | 26.97% | 0.883 | 27.634 | 1.140 | −0.225 | 0.851 |
| NVIDIA | LightGBM<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span> | GPU    | 4.1.0.99        |                             10% |               5m 56s |                    11s |          19m 10s | \$130,546 | \$74,296 | 27.27% | 0.884 | 27.929 | 1.143 | −0.231 | 0.846 |
| NVIDIA | XGBoost<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span>  | CPU    | 2.0.0.1         |                            100% |                1m 6s |                     4s |              20s | \$126,024 | \$73,961 | 26.76% | 0.886 | 27.467 | 1.130 | −0.209 | 0.867 |
| NVIDIA | XGBoost<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span>  | GPU    | 2.0.0.1         |                             92% |               1m 11s |                    13s |               4s | \$126,366 | \$73,981 | 26.86% | 0.885 | 27.543 | 1.131 | −0.214 | 0.865 |

### Hardware

### Tasks

### Input data and parameters

- populate input data

- Show params

- Tasks

  - Train using N sales, predict on test (perf numbers)
  - Time to train full set
  - Time to predict N rows
  - Time to calc N shaps

- Time savings calculation
