Model Benchmark
================

This repository compares the time-to-train of two gradient-boosted
decision tree (GBDT) frameworks - LightGBM and XGBoost - across
different hardware and versions. The purpose of this comparison is to
help the CCAO make two decisions:

1.  Which GBDT framework to use for its 2024 automated valuation models
2.  Whether or not to purchase additional hardware (a GPU) in order to
    improve model training speed

Below are the results of our tests.

> :warning: **NOTE:**
>
> Please note that the performance statistics presented here are only
> for cross-model comparison and do not reflect any real model results
> e.g. they are included only to show that each GBDT framework and
> version generates similar results, given the same data.

## Results

| Server | Type                                                                                                                         | Device | Package Version | Peak Device Utilization | Wall Time (Full Run) | Wall Time (Prediction) | Wall Time (SHAP) | RMSE      | MAE      | MAPE   | R2    | COD    | PRD   | PRB    | MKI   |
|:-------|:-----------------------------------------------------------------------------------------------------------------------------|:-------|:----------------|------------------------:|---------------------:|-----------------------:|-----------------:|:----------|:---------|:-------|:------|:-------|:------|:-------|:------|
| CCAO   | LightGBM                                                                                                                     | CPU    | 3.3.5           |                    100% |               3m 55s |                  1m 5s |        2h 1m 38s | \$130,989 | \$74,124 | 26.97% | 0.883 | 27.634 | 1.140 | −0.225 | 0.851 |
| CCAO   | LightGBM                                                                                                                     | CPU    | 4.1.0           |                    100% |                4m 8s |                  1m 4s |        2h 1m 48s | \$130,989 | \$74,124 | 26.97% | 0.883 | 27.634 | 1.140 | −0.225 | 0.851 |
| CCAO   | XGBoost<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span>  | CPU    | 2.0.0.1         |                    100% |               2m 41s |                    12s |           2m 14s | \$126,389 | \$74,129 | 26.76% | 0.885 | 27.458 | 1.130 | −0.212 | 0.866 |
| NVIDIA | LightGBM                                                                                                                     | CPU    | 3.3.5           |                    100% |               4m 12s |                     9s |           19m 5s | \$130,989 | \$74,124 | 26.97% | 0.883 | 27.634 | 1.140 | −0.225 | 0.851 |
| NVIDIA | LightGBM                                                                                                                     | CPU    | 4.1.0.99        |                    100% |                4m 7s |                     9s |          18m 52s | \$130,989 | \$74,124 | 26.97% | 0.883 | 27.634 | 1.140 | −0.225 | 0.851 |
| NVIDIA | LightGBM<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span> | GPU    | 4.1.0.99        |                     10% |               5m 56s |                    11s |          19m 10s | \$130,546 | \$74,296 | 27.27% | 0.884 | 27.929 | 1.143 | −0.231 | 0.846 |
| NVIDIA | XGBoost<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span>  | CPU    | 2.0.0.1         |                    100% |                1m 6s |                     4s |              20s | \$126,024 | \$73,961 | 26.76% | 0.886 | 27.467 | 1.130 | −0.209 | 0.867 |
| NVIDIA | XGBoost<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span>  | GPU    | 2.0.0.1         |                     92% |               1m 11s |                    13s |               4s | \$126,366 | \$73,981 | 26.86% | 0.885 | 27.543 | 1.131 | −0.214 | 0.865 |

1.  Categoricals with over 50 values are hashed, otherwise one-hot
    encoded.
2.  Categoricals with over 50 values are hashed, otherwise natively
    handled.

## Hardware

These tests were run on two different machines: an on-prem modeling
server used by the CCAO and a test server provided by NVIDIA via the
[LaunchPad](https://www.nvidia.com/en-us/launchpad/) program. The
machines have the following specifications:

|              | CCAO                                     | NVIDIA                                 |
|--------------|------------------------------------------|----------------------------------------|
| **CPU**      | Xeon Silver 4208 CPU @ 2.10GHz, 16 cores | Xeon Gold 6354 CPU @ 3.00GHz, 72 cores |
| **Memory**   | 128GiB                                   | 512GiB                                 |
| **GPU**      | \-                                       | NVIDIA A40, 48GB                       |
| **OS**       | Ubuntu 22.04 LTS                         | Ubuntu 22.04 LTS                       |
| **Compiler** | gcc (11.4.0) -O3 -march=native           | gcc (11.4.0) -O3 -march=native         |

## Tasks

The tasks performed for this benchmark (as shown in the results table)
are as follows:

- **Full Run** - The model is trained on the training + test set, to be
  used for prediction on unseen data.
- **Prediction** - The trained model is used to predict on the
  assessment data.
- **SHAP** - The trained model is used to predict SHAP values for the
  first 50K rows of assessment data.
- For performance metrics, the model is trained on a training set, then
  predicts on a holdout test set. Performance is calculated using the
  test set predictions.

## Inputs

The benchmark uses the following inputs:

- Input data from the 2023 CCAO residential valuation model:
  - [`input/training_data.parquet`](https://ccao-data-public-us-east-1.s3.amazonaws.com/models/inputs/res/2023/training_data.parquet) -
    424,950 rows
  - [`input/assessment_data.parquet`](https://ccao-data-public-us-east-1.s3.amazonaws.com/models/inputs/res/2023/assessment_data.parquet) -
    1,099,226 rows
- (Hyper)parameters used for this benchmark can be found in
  [`params.yaml`](./params.yaml)
