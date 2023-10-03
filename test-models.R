#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 1. Setup ---------------------------------------------------------------------
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
message("Setting up environment")

# Load libraries and scripts
options(tidymodels.dark = TRUE)
suppressPackageStartupMessages({
  library(arrow)
  library(ccao)
  library(dplyr)
  library(here)
  library(lightgbm)
  library(lightsnip)
  library(textrecipes)
  library(tictoc)
  library(tidymodels)
  library(vctrs)
  library(yaml)
})

# Load the parameters file containing the benchmark settings
params <- read_yaml("params.yaml")

# Get the number of available physical cores to use for multi-threading
num_threads <- parallel::detectCores(logical = FALSE)
set.seed(params$model$seed)




#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 2. Prepare Data --------------------------------------------------------------
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
message("Preparing model data")

# Load the full set of training data, then arrange by sale date in order to
# facilitate out-of-time sampling/validation
training_data_full <- read_parquet(here("input/training_data.parquet")) %>%
  filter(!ind_pin_is_multicard, !sv_is_outlier) %>%
  arrange(meta_sale_date)

# Create train/test split by time, with most recent observations in the test set
split_data <- initial_time_split(
  data = training_data_full,
  prop = params$cv$split_prop
)
test <- testing(split_data)
train <- training(split_data)

# Load the assessment data for prediction
assessment_data <- as_tibble(read_parquet("input/assessment_data.parquet"))




#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 3. LightGBM Models -----------------------------------------------------------
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
message("Initializing LightGBM models")


## 3.1. Initialize Models ------------------------------------------------------

# Gather lightgbm model-specific parameters
lgbm_params <- params$model$lightgbm

# Initialize a lightgbm model specification shared between different benchmarks
lgbm_model <- parsnip::boost_tree(
  stop_iter = NULL,
  trees = lgbm_params$parameter$num_iterations
) %>%
  set_mode("regression") %>%
  set_engine(
    engine = lgbm_params$engine,
    seed = params$model$seed,
    num_threads = num_threads,
    verbose = lgbm_params$verbose,

    objective = lgbm_params$objective,
    learning_rate = lgbm_params$parameter$learning_rate,
    categorical_feature = lgbm_params$predictor$categorical,

    validation = lgbm_params$parameter$validation_prop,
    sample_type = lgbm_params$parameter$validation_type,
    metric = lgbm_params$parameter$validation_metric,

    link_max_depth = lgbm_params$parameter$link_max_depth,
    max_bin = lgbm_params$parameter$max_bin,

    num_leaves = lgbm_params$hyperparameter$num_leaves,
    add_to_linked_depth = lgbm_params$hyperparameter$add_to_linked_depth,
    feature_fraction = lgbm_params$hyperparameter$feature_fraction,
    min_gain_to_split = lgbm_params$hyperparameter$min_gain_to_split,
    min_data_in_leaf = lgbm_params$hyperparameter$min_data_in_leaf,
    max_cat_threshold = lgbm_params$hyperparameter$max_cat_threshold,
    min_data_per_group = lgbm_params$hyperparameter$min_data_per_group,
    cat_smooth = lgbm_params$hyperparameter$cat_smooth,
    cat_l2 = lgbm_params$hyperparameter$cat_l2,
    lambda_l1 = lgbm_params$hyperparameter$lambda_l1,
    lambda_l2 = lgbm_params$hyperparameter$lambda_l2
  )

# Initialize a lightgbm recipe shared between models
lgbm_recipe <- recipe(training_data_full %>% select(-time_split)) %>%
  update_role(meta_sale_price, new_role = "outcome") %>%
  update_role(all_of(params$model$predictor$all), new_role = "predictor") %>%
  update_role(all_of(params$model$predictor$id), new_role = "ID") %>%
  update_role_requirements("ID", bake = FALSE) %>%
  update_role_requirements("NA", bake = FALSE) %>%
  step_rm(-all_outcomes(), -all_predictors())


## 3.2. CPU Model --------------------------------------------------------------

# Set the ID for capturing CPU model artifacts
lgbm_run_id_cpu <- paste0(params$machine, "-", "lightgbm-cpu")

# Create a CPU-model-specific recipe that converts categoricals to integers but
# otherwise keeps them untouched
lgbm_recipe_cpu <- lgbm_recipe %>%
  step_novel(all_of(params$model$predictor$categorical), -has_role("ID")) %>%
  step_unknown(all_of(params$model$predictor$categorical), -has_role("ID")) %>%
  step_integer(
    all_of(params$model$predictor$categorical), -has_role("ID"),
    strict = TRUE, zero_based = TRUE
  )

# Create a CPU-model-specific workflow to use for fitting and prediction
lgbm_wflow_cpu <- workflow() %>%
  add_model(lgbm_model) %>%
  add_recipe(
    recipe = lgbm_recipe_cpu,
    blueprint = hardhat::default_recipe_blueprint(allow_novel_levels = TRUE)
  )

# Fit model on training data and evaluate performance on holdout set
message("Fitting LightGBM CPU model on training data")
lgbm_wflow_cpu_fit <- fit(lgbm_wflow_cpu, data = train)
test %>%
  select(price = meta_sale_price) %>%
  mutate(pred = predict(lgbm_wflow_cpu_fit, test)$.pred) %>%
  summarize(
    id = lgbm_run_id_cpu,
    time = lubridate::now(),
    rmse = rmse_vec(price, pred),
    mae = mae_vec(price, pred),
    mape = mape_vec(price, pred),
    rsq = rsq_vec(price, pred),
    cod = assessr::cod(pred / price),
    prd = assessr::prd(pred, price),
    prb = assessr::prb(pred, price),
    mki = assessr::mki(pred, price)
  ) %>%
  arrow::write_parquet(file.path(
    "output/performance",
    paste0(lgbm_run_id_cpu, ".parquet")
  ))

# Fit the full data and gather timings
tictoc::tic.clearlog()
tictoc::tic(paste0(lgbm_run_id_cpu, "_train"))
message("Fitting LightGBM CPU model on full data")
lgbm_wflow_cpu_full_fit <- fit(lgbm_wflow_cpu, data = training_data_full)
tictoc::toc(log = TRUE)

# Load and prep the assessment data for prediction
assessment_data_prepped_cpu <- bake(
  object = lgbm_wflow_cpu_full_fit %>% extract_recipe(),
  new_data = assessment_data,
  all_predictors()
)

# Predict on the assessment data and gather timings
tictoc::tic(paste0(lgbm_run_id_cpu, "_predict"))
predict(
  lgbm_wflow_cpu_full_fit %>% extract_fit_parsnip(),
  new_data = assessment_data_prepped_cpu
)
tictoc::toc(log = TRUE)

# Predict 1K SHAP values and gather timings
tictoc::tic(paste0(lgbm_run_id_cpu, "_shap"))
predict(
  lgbm_wflow_cpu_full_fit %>% extract_fit_engine(),
  data = assessment_data_prepped_cpu %>% dplyr::slice(1:1000) %>% as.matrix(),
  predcontrib = TRUE
)
tictoc::toc(log = TRUE)

# Save CPU run timing logs to file
bind_rows(tictoc::tic.log(format = FALSE)) %>%
  arrow::write_parquet(file.path(
    "output/timing",
    paste0(lgbm_run_id_cpu, ".parquet")
  ))


## 3.3. GPU Model --------------------------------------------------------------

# Set the ID for capturing GPU model artifacts
lgbm_run_id_gpu <- paste0(params$machine, "-", "lightgbm-gpu")

# Create a list of categoricals to hash for GPU compatibility
lgbm_gpu_dummy_cats <- c(
  "meta_nbhd_code",
  "loc_cook_municipality_name",
  "loc_school_elementary_district_geoid",
  "loc_school_secondary_district_geoid"
)

# Create a GPU-model-specific recipe that randomly distributes categoricals
# into separate pools. This is to get around lightgbm GPU specific bugs related
# to the maximum bin size for categoricals
lgbm_recipe_gpu <- lgbm_recipe %>%
  step_novel(all_of(params$model$predictor$categorical), -has_role("ID")) %>%
  step_unknown(all_of(params$model$predictor$categorical), -has_role("ID")) %>%
  textrecipes::step_dummy_hash(all_of(lgbm_gpu_dummy_cats), num_terms = 24) %>%
  step_integer(
    all_of(params$model$predictor$categorical),
    -all_of("ID"), -all_of(lgbm_gpu_dummy_cats),
    strict = TRUE, zero_based = TRUE
  )

# Create a GPU-model-specific workflow to use for fitting and prediction. This
# sets the device and removed the hashed categoricals from the list of input
# parameters
lgbm_wflow_gpu <- workflow() %>%
  add_model(lgbm_model) %>%
  add_recipe(
    recipe = lgbm_recipe_gpu,
    blueprint = hardhat::default_recipe_blueprint(allow_novel_levels = TRUE)
  ) %>%
  set_args(
    device = "gpu",
    categorical_feature = setdiff(
      lgbm_params$predictor$categorical,
      lgbm_gpu_dummy_cats
    )
  )

# Fit model on training data and evaluate performance on holdout set
message("Fitting LightGBM GPU model on training data")
lgbm_wflow_gpu_fit <- fit(lgbm_wflow_gpu, data = train)
test %>%
  select(price = meta_sale_price) %>%
  mutate(pred = predict(lgbm_wflow_gpu_fit, test)$.pred) %>%
  summarize(
    id = lgbm_run_id_gpu,
    time = lubridate::now(),
    rmse = rmse_vec(price, pred),
    mae = mae_vec(price, pred),
    mape = mape_vec(price, pred),
    rsq = rsq_vec(price, pred),
    cod = assessr::cod(pred / price),
    prd = assessr::prd(pred, price),
    prb = assessr::prb(pred, price),
    mki = assessr::mki(pred, price)
  ) %>%
  arrow::write_parquet(file.path(
    "output/performance",
    paste0(lgbm_run_id_gpu, ".parquet")
  ))

# Fit the full data and gather timings
tictoc::tic.clearlog()
tictoc::tic(paste0(lgbm_run_id_gpu, "_train"))
message("Fitting LightGBM GPU model on full data")
lgbm_wflow_gpu_full_fit <- fit(lgbm_wflow_gpu, data = training_data_full)
tictoc::toc(log = TRUE)

# Load and prep the assessment data for prediction
assessment_data_prepped_gpu <- bake(
  object = lgbm_wflow_gpu_full_fit %>% extract_recipe(),
  new_data = assessment_data,
  all_predictors()
)

# Predict on the assessment data and gather timings
tictoc::tic(paste0(lgbm_run_id_gpu, "_predict"))
predict(
  lgbm_wflow_gpu_full_fit %>% extract_fit_parsnip(),
  new_data = assessment_data_prepped_gpu
)
tictoc::toc(log = TRUE)

# Predict 1K SHAP values and gather timings
tictoc::tic(paste0(lgbm_run_id_gpu, "_shap"))
predict(
  lgbm_wflow_gpu_full_fit %>% extract_fit_engine(),
  data = assessment_data_prepped_gpu %>% dplyr::slice(1:1000) %>% as.matrix(),
  predcontrib = TRUE
)
tictoc::toc(log = TRUE)

# Save gpu run timing logs to file
bind_rows(tictoc::tic.log(format = FALSE)) %>%
  arrow::write_parquet(file.path(
    "output/timing",
    paste0(lgbm_run_id_gpu, ".parquet")
  ))
