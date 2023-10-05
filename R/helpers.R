get_timings <- function(machine,
                        model_type,
                        device_type,
                        workflow,
                        recipe,
                        training_data,
                        test_data,
                        full_data,
                        assessment_data,
                        n_shap = 1000
                        ) {

  model_version <- packageVersion(model_type)
  run_id = glue("{machine}-{model_type}-{model_version}-{device_type}")

  # Fit model on training data and evaluate performance on holdout set
  message(glue("Fitting {run_id} on training data"))
  wflow_train_fit <- parsnip::fit(workflow, data = training_data)

  test_data %>%
    select(price = meta_sale_price) %>%
    mutate(pred = predict(wflow_train_fit, test_data)$.pred) %>%
    summarize(
      run_id = run_id,
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
    arrow::write_parquet(glue("output/performance/{run_id}.parquet"))

  # Fit the full data and gather timings
  message(glue("Fitting {run_id} on full data"))
  tictoc::tic.clearlog()
  tictoc::tic(glue("{run_id}_train"))
  wflow_full_fit <- parsnip::fit(workflow, data = full_data)
  tictoc::toc(log = TRUE)

  # Load and prep the assessment data for prediction
  assessment_data_prepped <- bake(
    object = wflow_full_fit %>% extract_recipe(),
    new_data = assessment_data,
    all_predictors()
  )

  # Predict on the assessment data and gather timings
  message(glue("Predicting on assessment data {run_id}"))
  tictoc::tic(glue("{run_id}_predict"))
  predict(
    wflow_full_fit %>% extract_fit_parsnip(),
    new_data = assessment_data_prepped
  )
  tictoc::toc(log = TRUE)

  # Predict SHAP values and gather timings
  message(glue("Predicting SHAPs for {run_id}"))
  tictoc::tic(glue("{run_id}_shap"))

  wflow_eng_fit <- wflow_full_fit %>% extract_fit_engine()
  shap_slice <- assessment_data_prepped %>%
    dplyr::slice(seq_len(n_shap)) %>%
    as.matrix()

  if (model_type == "lightgbm" & model_version > "3.3.5") {
    predict(wflow_eng_fit, shap_slice, type = "contrib") %>%
      as_tibble(.name_repair = "unique") %>%
      arrow::write_parquet(glue("output/shap/{run_id}.parquet"))
  } else {
    predict(wflow_eng_fit, shap_slice, predcontrib = TRUE) %>%
      as_tibble(.name_repair = "unique") %>%
      arrow::write_parquet(glue("output/shap/{run_id}.parquet"))
  }

  tictoc::toc(log = TRUE)

  # Save run timing logs to file
  bind_rows(tictoc::tic.log(format = FALSE)) %>%
    select(- callback_msg) %>%
    mutate(time = toc - tic) %>%
    tidyr::separate(msg, into = c("run_id", "stage"), sep = "_") %>%
    arrow::write_parquet(glue("output/timing/{run_id}.parquet"))
}
