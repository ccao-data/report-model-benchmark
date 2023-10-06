Model Benchmark
================

This repository compares the time-to-train of two gradient-boosted
decision tree (GBDT) frameworks - LightGBM and XGBoost - across
different hardware and versions. The purpose of this comparison is to
help the CCAO make two decisions:

1.  Which GBDT framework to use for its 2024 automated valuation models
2.  Whether or not to purchase or rent additional hardware (a GPU) in
    order to improve model training speed

Below are the results of our tests:

> :warning: **NOTE:**
>
> The performance statistics presented here are only for cross-model
> comparison and do not reflect any real model results e.g. they are
> included only to show that each GBDT framework and version generates
> similar results, given the same data.

## Results

| Server | Type                                                                                                                         | Device | Package Version | Peak Device Utilization | Wall Time (Full Run) | Wall Time (Prediction) | Wall Time (SHAP) |      RMSE |      MAE |   MAPE |    R2 |    COD |   PRD | PRB    | MKI   |
|:-------|:-----------------------------------------------------------------------------------------------------------------------------|:-------|:----------------|------------------------:|---------------------:|-----------------------:|-----------------:|----------:|---------:|-------:|------:|-------:|------:|:-------|:------|
| CCAO   | LightGBM                                                                                                                     | CPU    | 3.3.5           |                    100% |               3m 55s |                  1m 5s |        2h 1m 38s | \$130,989 | \$74,124 | 26.97% | 0.883 | 27.634 | 1.140 | −0.225 | 0.851 |
| CCAO   | LightGBM                                                                                                                     | CPU    | 4.1.0           |                    100% |                4m 8s |                  1m 4s |        2h 1m 48s | \$130,989 | \$74,124 | 26.97% | 0.883 | 27.634 | 1.140 | −0.225 | 0.851 |
| CCAO   | XGBoost<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span>  | CPU    | 2.0.0.1         |                    100% |               2m 41s |                    12s |           2m 14s | \$126,389 | \$74,129 | 26.76% | 0.885 | 27.458 | 1.130 | −0.212 | 0.866 |
| NVIDIA | LightGBM                                                                                                                     | CPU    | 3.3.5           |                    100% |               4m 12s |                     9s |           19m 5s | \$130,989 | \$74,124 | 26.97% | 0.883 | 27.634 | 1.140 | −0.225 | 0.851 |
| NVIDIA | LightGBM                                                                                                                     | CPU    | 4.1.0.99        |                    100% |               1m 37s |                    17s |          19m 44s | \$130,989 | \$74,124 | 26.97% | 0.883 | 27.634 | 1.140 | −0.225 | 0.851 |
| NVIDIA | LightGBM<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span> | GPU    | 4.1.0.99        |                     10% |               2m 33s |                    19s |          19m 39s | \$130,541 | \$74,282 | 27.23% | 0.884 | 27.903 | 1.143 | −0.230 | 0.847 |
| NVIDIA | XGBoost<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span>  | CPU    | 2.0.0.1         |                    100% |               1m 25s |                     5s |              52s | \$126,894 | \$74,191 | 26.83% | 0.884 | 27.552 | 1.130 | −0.210 | 0.868 |
| NVIDIA | XGBoost<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span>  | GPU    | 2.0.0.1         |                     92% |               1m 12s |                    13s |               5s | \$126,952 | \$74,124 | 26.78% | 0.885 | 27.515 | 1.130 | −0.213 | 0.867 |
| AWS    | LightGBM                                                                                                                     | CPU    | 4.1.0.99        |                    100% |               2m 38s |                  1m 1s |       1h 45m 59s | \$130,989 | \$74,124 | 26.97% | 0.883 | 27.634 | 1.140 | −0.225 | 0.851 |
| AWS    | LightGBM<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span> | GPU    | 4.1.0.99        |                      8% |               3m 56s |                 1m 10s |       1h 47m 19s | \$130,632 | \$74,210 | 27.18% | 0.884 | 27.866 | 1.143 | −0.230 | 0.847 |
| AWS    | XGBoost<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span>  | CPU    | 2.0.0.1         |                    100% |               1m 58s |                    11s |           1m 31s | \$125,867 | \$73,899 | 26.79% | 0.886 | 27.510 | 1.130 | −0.212 | 0.869 |
| AWS    | XGBoost<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span>  | GPU    | 2.0.0.1         |                     95% |               1m 31s |                    17s |               6s | \$126,679 | \$74,002 | 26.80% | 0.885 | 27.523 | 1.130 | −0.210 | 0.867 |

1.  Categoricals with over 50 values are hashed, otherwise one-hot
    encoded.
2.  Categoricals with over 50 values are hashed, otherwise natively
    handled.

## Cost estimates

| Server                                                                                                                     | Type     | Device | Instance Type | Est. Time Per Run | Est. Cost Per Run | Est. Total 2024 Cost |
|:---------------------------------------------------------------------------------------------------------------------------|:---------|:-------|:--------------|------------------:|------------------:|---------------------:|
| CCAO                                                                                                                       | LightGBM | CPU    | \-            |        57h 55m 5s |                \- |                   \- |
| CCAO                                                                                                                       | XGBoost  | CPU    | \-            |        9h 25m 14s |                \- |                   \- |
| NVIDIA<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span> | XGBoost  | GPU    | \-            |        3h 54m 34s |           \$35.00 |           \$7,000.00 |
| AWS<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span>    | XGBoost  | GPU    | Normal        |        4h 54m 10s |            \$7.96 |           \$1,592.48 |
| AWS<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span>    | XGBoost  | GPU    | Spot          |        4h 54m 10s |            \$3.29 |             \$658.56 |

1.  Estimate assumes a fixed cost for an NVIDIA A40 of \$7,000.
2.  Estimates use AWS costs for `g5.4xlarge` instances created
    ephemerally using AWS Batch + Fargate. As of 2023-10-06, costs are:
    - Normal hourly pricing: \$1.62
    - Spot hourly pricing: \$0.67
3.  All estimates assume 200 total runs in 2024, costs per run decrease
    for the NVIDIA option as number of runs increases.

## Hardware

These tests were run on three different machines: an on-prem modeling
server used by the CCAO, a test server provided by NVIDIA via the
[LaunchPad](https://www.nvidia.com/en-us/launchpad/) program, and a
standard AWS `g5.4xlarge` EC2 instance. The machines have the following
specifications:

|              | CCAO                                     | NVIDIA                                 | g5.4xlarge                       |
|--------------|------------------------------------------|----------------------------------------|----------------------------------|
| **CPU**      | Xeon Silver 4208 CPU @ 2.10GHz, 16 cores | Xeon Gold 6354 CPU @ 3.00GHz, 16 cores | AMD EPYC 7R32 @ 3.3GHz, 16 cores |
| **Memory**   | 128GiB                                   | 512GiB                                 | 64GiB                            |
| **GPU**      | \-                                       | NVIDIA A40, 48GB                       | NVIDIA A10G, 24GB                |
| **OS**       | Ubuntu 22.04 LTS                         | Ubuntu 22.04 LTS                       | Ubuntu 22.04 LTS                 |
| **Compiler** | gcc (11.4.0) -O3 -march=native           | gcc (11.4.0) -O3 -march=native         | gcc (11.4.0) -O3 -march=native   |

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
