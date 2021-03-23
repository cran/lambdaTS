#' lambdaTS
#'
#' @param data A data frame with ts on columns and possibly a date column (not mandatory)
#' @param target String. Time series names to be jointly analyzed within the seq2seq model
#' @param future Positive integer. The future dimension with number of time-steps to be predicted
#' @param past Positive integer. The past dimension with number of time-steps in the past used for the prediction. Default: future
#' @param ci Confidence interval. Default: 0.8
#' @param deriv Positive integer. Number of differentiation operations to perform on the original series. 0 = no change; 1: one diff; 2: two diff, and so on.
#' @param yjt Logical. Performing Yeo-Johnson Transformation on data is always advisable, especially when dealing with different ts at different scales. Default: TRUE
#' @param shift Vector of positive integers. Allow for target variables to shift ahead of time. Zero means no shift. Length must be equal to the number of targets.
#' @param smoother Logical. Perform optimal smooting using standard loess. Default: FALSE
#' @param k_embed Positive integer. Number of Time2Vec embedding dimensions. Minimum value is 2. Default: 30
#' @param r_proj Positive integer. Number of dimensions for the reduction space (to reduce quadratic complexity). Must be largely less than k_embed size. Default: ceiling(k_embed/3) + 1
#' @param n_heads Positive integer. Number of heads for the attention mechanism. Computationally expensive, use with care. Default: 1
#' @param n_bases Positive integer. Number of normal curves to build on each parameter. WIth more than one base you can model be Default: 1
#' @param activ String. The activation function for the linear transformation of the attention matrix into the future sequence. Implemented options are: "linear", "leaky_relu", "celu", "elu", "gelu", "selu", "softplus", "bent", "snake", "softmax", "softmin", "softsign", "sigmoid", "tanh", "tanhshrink", "swish", "hardtanh", "mish". Default: "linear".
#' @param loss_metric String. Loss function for the variational model. Two options: "elbo" or "crps". Default: "crps".
#' @param optim String. Optimization methods available are: "adadelta", "adagrad", "rmsprop", "rprop", "sgd", "asgd", "adam". Default: "adam".
#' @param epochs Positive integer. Default: 30.
#' @param lr Positive numeric. Learning rate. Default: 0.01.
#' @param patience Positive integer. Waiting time (in epochs) before evaluating the overfit performance.
#' @param verbose Logical. Default: TRUE
#' @param sample_n Positive integer. Number of samples from the variational model to evalute the mean forecast values. Computationally expensive, use with care. Default: 100.
#' @param seed Random seed. Default: 42.
#' @param dev String. Torch implementation of computational platform: "cpu" or "cuda" (gpu). Default: "cpu".
#' @param starting_date Date. Initial date to assign temporal values to the series. Default: NULL (progressive numbers).
#' @param dbreak String. Minimum time marker for x-axis, in liberal form: i.e., "3 months", "1 week", "20 days". Default: NULL.
#' @param days_off String. Weekdays to exclude (i.e., c("saturday", "sunday")). Default: NULL.
#' @param min_set Positive integer. Minimun number for validation set in case of automatic resize of past dimension. Default: future.
#' @param holdout Positive numeric. Percentage of time series for holdout validation. Default: 0.5.
#' @param batch_size Positive integer. Default: 30.
#'
#' @author Giancarlo Vercellino \email{giancarlo.vercellino@gmail.com}
#'
#' @return This function returns a list including:
#' \itemize{
#' \item prediction: a table with quantile predictions, mean and std for each ts
#' \item history: plot of loss during the training process for the joint-transformed ts
#' \item plot: graph with history and prediction for each ts
#' \item learning_error: errors for the joint-transformed ts (rmse, mae, mdae, mpe, mape, smape, rrse, rae)
#' \item feature_errors: errors for each ts (rmse, mae, mdae, mpe, mape, smape, rrse, rae)
#' \item pred_stats: some stats on predicted ts (average iqr, iqr ratio t/1, approx upside probability)
#' \item time_log
#' }
#'
#' @export
#'
#' @importFrom fANCOVA loess.as
#' @importFrom imputeTS na_kalman
#' @importFrom bestNormalize yeojohnson
#' @importFrom modeest mlv1
#' @import purrr
#' @import abind
#' @import torch
#' @import ggplot2
#' @import tictoc
#' @import readr
#' @import stringr
#' @import lubridate
#' @importFrom scales number
#' @importFrom narray split
#' @importFrom stats lm median na.omit quantile
#' @importFrom utils head tail
#' @importFrom bizdays create.calendar add.bizdays bizseq
#'
#'@examples
#'\dontrun{
#'lambdaTS(bitcoin_gold_oil, c("bitcoin_Close", "gold_close", "oil_Close"), 30, 30, deriv = 2)
#'}
#'
lambdaTS <- function(data, target, future, past = future, ci = 0.8, deriv = 1, yjt = TRUE, shift = 0, smoother = FALSE,
                     k_embed = 30, r_proj = ceiling(k_embed/3) + 1, n_heads = 1, n_bases = 1, activ = "linear", loss_metric = "elbo", optim = "adam",
                     epochs = 30, lr = 0.01, patience = epochs, verbose = TRUE, sample_n = 100, seed = 42, dev ="cpu",
                     starting_date = NULL, dbreak = NULL, days_off = NULL, min_set = future, holdout = 0.5, batch_size = 30)
{
  tic.clearlog()
  tic("time")

  ###VARIATIONAL SEQ2SEQ LAMBDA TRANSFORMER MODEL FOR TIME SERIES ANALYSIS



  ###PRECHECK
  if(deriv >= future){stop("deriv cannot be equal or greater than future")}
  if(future <= 0 | past <= 0){stop("past and future must be strictly positive integers")}
  if(k_embed < 2){k_embed <- 2; message("setting k_embed to the minimum possible value (2)\n")}
  if(r_proj >= k_embed){r_proj <- ceiling(k_embed/3) + 1; message("reducing r_proj to a third of k_embed\n")}


  ####PREPARATION
  set.seed(seed)
  torch_manual_seed(seed)

  data <- data[, target, drop = FALSE]
  n_feat <- ncol(data)
  n_length <- nrow(data)

  ###MISSING IMPUTATION
  if(anyNA(data) & is.null(days_off)){data <- as.data.frame(map(data, ~ na_kalman(.x))); message("kalman imputation\n")}
  if(anyNA(data) & !is.null(days_off)){data <- na.omit(data); message("omitting missings\n")}

  ###SHIFT
  if(any(shift > 0) & length(shift)==n_feat)
  {
    data <- map2(data, shift, ~ tail(.x, n_length - .y))
    n_length <- min(map_dbl(data, ~ length(.x)))
    data <- map_df(data, ~ head(.x, n_length))
  }

  ###SMOOTHING
  if(smoother==TRUE){data <- as.data.frame(map(data, ~ loess.as(x=1:n_length, y=.x)$fitted)); message("performing optimal smoothing\n")}

  ###SEGMENTATION
  orig <- data ###FOR REFERENCE, SINCE DATA MAY BE TRANSFORMED
  train_index <- 1:floor(n_length * (1-holdout))
  val_index <- setdiff(1:n_length, train_index)

  if((length(train_index) + min_set - 1) < (past + future) | (length(val_index) + min_set - 1) < (past + future))
  {
    message("insufficient data points for the forecasting horizon\n")
    past <- min(c(length(train_index), length(val_index))) - future - (min_set - 1)
    if(past >= 1){message("setting past to max possible value,", past,"points, for a minimal", min_set ,"validation sequences\n")}
    if(past < 1){stop("need to reset both past and future parameters or adjust holdout\n")}
  }

  feature_block <- 1:(past + deriv)
  target_block <- (past - deriv + 1):(past + future)##DERIV OVERLAP ZONE

  train_reframed <- reframe(data[train_index,,drop=FALSE], past + future)
  if(anyNA(train_reframed)){stop("missings still present in train set")}
  x_train <- train_reframed[,feature_block,,drop=FALSE]
  y_train <- train_reframed[,target_block,,drop=FALSE]

  val_reframed <- reframe(data[val_index,,drop=FALSE], past + future)
  if(anyNA(val_reframed)){stop("missings still present in validation set")}
  x_val <- val_reframed[,feature_block,,drop=FALSE]
  y_val <- val_reframed[,target_block,,drop=FALSE]

  new_data <- tail(data, past + deriv)
  new_reframed <- reframe(new_data, past + deriv)

  ###DERIVATIVE
  if(length(deriv) > 1){stop("require a single deriv value")}
  if(deriv > 0)
  {
    x_train_deriv_model <- reframed_differentiation(x_train, deriv)
    x_train <- x_train_deriv_model$difframed
    y_train_deriv_model <- reframed_differentiation(y_train, deriv)
    y_train <- y_train_deriv_model$difframed

    x_val_deriv_model <- reframed_differentiation(x_val, deriv)
    x_val <- x_val_deriv_model$difframed
    y_val_deriv_model <- reframed_differentiation(y_val, deriv)
    y_val <- y_val_deriv_model$difframed

    new_data_deriv_model <- reframed_differentiation(new_reframed, deriv)
    new_reframed <- new_data_deriv_model$difframed
  }

  ###YJ TRANSFORMATION
  if(yjt==TRUE)
  {
    melt <- map(smart_split(x_train, along = 3), ~ unique(as.vector(.x)))
    yj_pars <-  map(melt, ~ yeojohnson(.x))

    x_train <- abind(map2(yj_pars, smart_split(x_train, along = 3), ~ predict(.x, .y)), along = 3)
    y_train <- abind(map2(yj_pars, smart_split(y_train, along = 3), ~ predict(.x, .y)), along = 3)
    x_val <- abind(map2(yj_pars, smart_split(x_val, along = 3), ~ predict(.x, .y)), along = 3)
    y_val <- abind(map2(yj_pars, smart_split(y_val, along = 3), ~ predict(.x, .y)), along = 3)
    new_reframed <- abind(map2(yj_pars, smart_split(new_reframed, along = 3), ~ array(predict(.x, .y), dim = c(1,length(.y),1))), along = 3)

    message("yeo-johnson transformation\n")
    if(anyNA(x_train)|anyNA(y_train)|anyNA(x_val)|anyNA(y_val)|anyNA(new_reframed)){stop("error in yeo-johnson transformation")}
  }

  ###TRAINING MODEL
  model <- nn_variational_model(target_len = future, seq_len = past, k_embed = k_embed, r_proj = r_proj, n_heads = n_heads, n_bases = n_bases, activ = activ, dev = dev)
  training <- training_function(model, x_train, y_train, x_val, y_val, loss_metric, optim, lr, epochs, patience, verbose, batch_size, dev)
  train_history <- training$train_history
  val_history <- training$val_history
  model <- training$model

  pred_train <- pred_fun(model, x_train, "mean", dev = dev, sample_n = sample_n)
  pred_val <- pred_fun(model, x_val, "mean", dev =dev, sample_n = sample_n)
  samp_val <- replicate(sample_n, pred_fun(model, x_val, "sample", dev = dev, sample_n = sample_n), simplify = FALSE)
  train_errors <- eval_metrics(y_train, pred_train)
  val_errors <- eval_metrics(y_val, pred_val)
  learning_error <- Reduce(rbind, list(train_errors, val_errors))
  rownames(learning_error) <- c("training", "validation")

  ###NEW PREDICTION
  pred_new <- replicate(sample_n, pred_fun(model, new_reframed, "sample", dev = dev, sample_n = sample_n), simplify = FALSE)

  ###INTEGRATION & YJ BACK-TRANSFORM
  if(yjt==TRUE)
  {
    pred_train <- abind(map2(yj_pars, smart_split(pred_train, along = 3), ~ predict(.x, newdata = .y, inverse = TRUE)), along = 3)
    pred_val <- abind(map2(yj_pars, smart_split(pred_val, along = 3), ~ predict(.x, newdata = .y, inverse = TRUE)), along = 3)
    samp_val <- map(samp_val, ~ abind(map2(yj_pars, smart_split(.x, along = 3), ~ predict(.x, newdata = .y, inverse = TRUE)), along = 3))
    pred_new <- map(pred_new, ~ abind(map2(yj_pars, smart_split(.x, along = 3), ~ array(predict(.x, newdata = .y, inverse = TRUE), dim = c(1, future, 1))), along = 3))
  }

  if(deriv > 0)
  {
    pred_train <- reframed_integration(pred_train, y_train_deriv_model$heads, deriv)
    pred_val <- reframed_integration(pred_val, y_val_deriv_model$heads, deriv)
    samp_val <- map(samp_val, ~ reframed_integration(.x, y_val_deriv_model$heads, deriv))
    pred_new <- map(pred_new, ~ reframed_integration(.x, new_data_deriv_model$tails, deriv))###UNLIST NEEDED ONLY HERE
  }

  ###ERRORS
  predict_block <- (past + 1):(past + future)
  train_true <- train_reframed[,predict_block,,drop=FALSE]
  train_errors <- map2(smart_split(train_true, along = 3), smart_split(pred_train, along = 3), ~ eval_metrics(.x, .y))

  predict_block <- (past + 1):(past + future)
  val_true <- val_reframed[,predict_block,,drop=FALSE]
  val_errors <- map2(smart_split(val_true, along = 3), smart_split(pred_val, along = 3), ~ eval_metrics(.x, .y))

  features_errors <- transpose(list(train_errors, val_errors))
  features_errors <- map(features_errors, ~ Reduce(rbind, .x))
  features_errors <- map(features_errors, ~ {rownames(.x) <- c("training", "validation"); return(.x)})
  names(features_errors) <- target

  mean_raw_errors <- aperm(apply(abind(map(samp_val, ~ val_true - .x), along = 4), c(1, 2, 3), mean, na.rm=TRUE), c(2, 1, 3))
  mean_pred_new <- abind(replicate(dim(mean_raw_errors)[2] , aperm(apply(abind(pred_new, along=4), c(1, 2, 3), mean), c(2, 1, 3)), simplify = FALSE), along = 2)

  quants <- sort(unique(c((1-ci)/2, 0.25, 0.5, 0.75, ci+(1-ci)/2)))
  q_names <- paste0("q", quants * 100)

  integrated_pred <- mean_raw_errors + mean_pred_new
  integrated_pred <- smart_split(integrated_pred, along = 3)

  domain_check <- as_mapper(~ ifelse(all(.x < 0), "neg", ifelse(all(.x > 0), "pos", "mix")))
  domains <- map(orig, domain_check)

  keep_index <- map2(integrated_pred, domains, ~ suppressWarnings(apply(.x, 2, function(x) switch(.y, "pos" = all(x > 0) & all(is.finite(x)), "neg" = all(x < 0) & all(is.finite(x)), "mix" = all(is.finite(x))))))
  integrated_pred <- map2(integrated_pred, keep_index, ~ .x[ , .y, drop=FALSE])

  pred_quantiles <- map(integrated_pred, ~ t(apply(.x, 1, function(x) {round(c(quantile(x, probs = quants, na.rm = TRUE), mean(x, na.rm=TRUE), sd(x, na.rm=TRUE)), 3)})))
  prediction <- map(pred_quantiles, ~{rownames(.x) <- paste0("t", 1:future); colnames(.x) <- c(q_names, "mean", "sd"); return(.x)})
  names(prediction) <- target

  ###PREDICTION STATISTICS
  pred_stats <- pred_statistics(prediction, tail(orig, 1), future, target)

  ###TRAINING PLOT
  act_epochs <- length(train_history)
  x_ref_point <- c(which.min(val_history), which.min(train_history))
  y_ref_point <- c(min(val_history), min(train_history))

  train_data <- data.frame(epochs = 1:act_epochs, train_loss = train_history)

  history <- ggplot(train_data) + geom_point(aes(x = epochs, y = train_loss), col = "blue", shape = 1, size = 1)
  if(act_epochs > 5){history <- history + geom_smooth(col="darkblue", aes(x = epochs, y = train_loss), se=FALSE, method = "loess")}

  val_data <- data.frame(epochs = 1:act_epochs, val_loss = val_history)

  history <- history + geom_point(aes(x = epochs, y = val_loss), val_data, col = "orange", shape = 1, size = 1)
  if(act_epochs > 5){history <- history + geom_smooth(aes(x = epochs, y = val_loss), val_data, col="darkorange", se=FALSE, method = "loess")}

  if(loss_metric == "elbo"){history <- history + scale_y_log10() + ylab("mae + absolute elbo (log10 scale)")}
  if(loss_metric == "crps"){history <- history + ylab("crps")}

  history <- history + xlab("epochs") +
    annotate("text", x = x_ref_point, y = y_ref_point, label = c("VALIDATION SET", "TRAINING SET"),
             col = c("darkorange", "darkblue"), hjust = 0, vjust= 0)

  ###SETTING DATES
  if(!is.null(starting_date))
  {
    date_list <- map(shift, ~ seq.Date(as.Date(starting_date) + .x, as.Date(starting_date) + .x + n_length, length.out = n_length))
    dates <- date_list
    start_date <- map(date_list, ~ tail(.x, 1))
    mycal <- create.calendar(name="mycal", weekdays=days_off)
    end_day <- map(start_date, ~ add.bizdays(.x, future, cal=mycal))
    pred_dates <- map2(start_date, end_day, ~ tail(bizseq(.x, .y, cal=mycal), future))
    prediction <- map2(prediction, pred_dates, ~ as.data.frame(cbind(dates=.y, .x)))
    prediction <- map(prediction, ~ {.x$dates <- as.Date(.x$dates, origin = "1970-01-01"); return(.x)})
  }

  if(is.null(starting_date))
  {
    dates <- 1:n_length
    dates <- replicate(n_feat, dates, simplify = FALSE)
    pred_dates <- (n_length+1):(n_length+future)
    pred_dates <- replicate(n_feat, pred_dates, simplify = FALSE)
  }

  ###PREDICTION PLOT
  lower_name <- paste0("q", ((1-ci)/2) * 100)
  upper_name <- paste0("q", (ci+(1-ci)/2) * 100)

  plot <- pmap(list(orig, prediction, target, dates, pred_dates), ~ ts_graph(x_hist = ..4, y_hist = ..1, x_forcat = ..5, y_forcat = ..2[, "mean"],
                                                                             lower = ..2[, lower_name], upper = ..2[, upper_name], label_x = paste0("Seq2Seq Variational Lambda (past = ", past ,", future = ", future,")"),
                                                                             label_y = paste0(str_to_title(..3), " Values"), dbreak = dbreak))

  toc(log = TRUE)
  time_log<-seconds_to_period(round(parse_number(unlist(tic.log())), 0))

  ###OUTCOMES
  outcome <- list(prediction = prediction, history = history, plot = plot, learning_error = learning_error,
                  features_errors = features_errors, pred_stats = pred_stats, time_log = time_log)

  return(outcome)
}
