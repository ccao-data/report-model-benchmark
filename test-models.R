#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 1. Setup ---------------------------------------------------------------------
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
message("Setting up environment")

# Start the timer and clear logs from prior stage
tictoc::tic.clearlog()
tictoc::tic("Train")

# Load libraries and scripts
options(tidymodels.dark = TRUE)
suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(here)
  library(lightgbm)
  library(lightsnip)
  library(tictoc)
  library(tidymodels)
  library(vctrs)
  library(yaml)
})

# Load helpers and recipes from files
walk(list.files("R/", "\\.R$", full.names = TRUE), source)

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




#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 3. LightGBM Models -----------------------------------------------------------
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
message("Initializing LightGBM models")

## 3.1. Initialize Models ------------------------------------------------------

# Gather model-specific parameters
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


## 3.1. CPU Model --------------------------------------------------------------

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

message("Fitting LightGBM CPU model on training data")
lgbm_wflow_cpu_fit <- fit(lgbm_wflow_cpu, data = train)

message("Fitting LightGBM CPU model on full data")
lgbm_wflow_cpu_full_fit <- fit(lgbm_wflow_cpu, data = training_data_full)




#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 4. Finalize Models -----------------------------------------------------------
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
message("Finalizing and saving trained model")

# Get predictions on the test set using the training data model. These
# predictions are used to evaluate model performance on the unseen test set.
# Keep only the variables necessary for evaluation
test %>%
  mutate(pred_card_initial_fmv = predict(lgbm_wflow_final_fit, test)$.pred) %>%
  select(
    meta_year, meta_pin, meta_class, meta_card_num,
    meta_triad_code, meta_township_code, meta_nbhd_code,
    loc_cook_municipality_name, loc_ward_num, loc_census_puma_geoid,
    loc_census_tract_geoid, loc_school_elementary_district_geoid,
    loc_school_secondary_district_geoid, loc_school_unified_district_geoid,
    char_bldg_sf,
    all_of(c(
      "prior_far_tot" = params$ratio_study$far_column,
      "prior_near_tot" = params$ratio_study$near_column
    )),
    pred_card_initial_fmv,
    meta_sale_price, meta_sale_date, meta_sale_document_num
  ) %>%
  # Prior year values are AV, not FMV. Multiply by 10 to get FMV for residential
  mutate(
    prior_far_tot = prior_far_tot * 10,
    prior_near_tot = prior_near_tot * 10
  ) %>%
  as_tibble() %>%
  write_parquet(paths$output$test_card$local)

# Save the finalized model object to file so it can be used elsewhere. Note the
# lgbm_save() function, which uses lgb.save() rather than saveRDS(), since
# lightgbm is picky about how its model objects are stored on disk
lgbm_wflow_final_full_fit %>%
  workflows::extract_fit_parsnip() %>%
  lightsnip::lgbm_save(paths$output$workflow_fit$local)

# Save the finalized recipe object to file so it can be used to preprocess
# new data. This is critical since it saves the factor levels used to integer-
# encode any categorical columns
lgbm_wflow_final_full_fit %>%
  workflows::extract_recipe() %>%
  lightsnip::axe_recipe() %>%
  saveRDS(paths$output$workflow_recipe$local)

# End the stage timer and write the time elapsed to a temporary file
tictoc::toc(log = TRUE)
bind_rows(tictoc::tic.log(format = FALSE)) %>%
  arrow::write_parquet(gsub("//*", "/", file.path(
    paths$intermediate$timing$local,
    "model_timing_train.parquet"
  )))
