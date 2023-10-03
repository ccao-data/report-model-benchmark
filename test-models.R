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
  library(glue)
  library(here)
  library(lightgbm)
  library(lightsnip)
  library(text2vec)
  library(textrecipes)
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

# Run the benchmark workflow for CPU model
get_timings(
  machine = params$machine,
  model_type = "lightgbm",
  device_type = "cpu",
  workflow = lgbm_wflow_cpu,
  recipe = lgbm_recipe_cpu,
  training_data = train,
  test_data = test,
  full_data = training_data_full,
  assessment_data = assessment_data
)


## 3.3. GPU Model --------------------------------------------------------------

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
    all_of(setdiff(params$model$predictor$categorical, lgbm_gpu_dummy_cats)),
    strict = TRUE, zero_based = TRUE
  )

# Create a GPU-model-specific workflow to use for fitting and prediction. This
# sets the device and removed the hashed categoricals from the list of input
# parameters
lgbm_wflow_gpu <- workflow() %>%
  add_model(
    lgbm_model %>%
      set_args(
        device = "gpu",
        categorical_feature = setdiff(
          lgbm_params$predictor$categorical,
          lgbm_gpu_dummy_cats
        )
      )
  ) %>%
  add_recipe(
    recipe = lgbm_recipe_gpu,
    blueprint = hardhat::default_recipe_blueprint(allow_novel_levels = TRUE)
  )

# Run the benchmark workflow for GPU model
get_timings(
  machine = params$machine,
  model_type = "lightgbm",
  device_type = "gpu",
  workflow = lgbm_wflow_gpu,
  recipe = lgbm_recipe_gpu,
  training_data = train,
  test_data = test,
  full_data = training_data_full,
  assessment_data = assessment_data
)
