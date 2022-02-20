#' support functions for lambdaTS
#'
#' @author Giancarlo Vercellino \email{giancarlo.vercellino@gmail.com}
#'
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
#' @importFrom stats lm median na.omit quantile density
#' @importFrom utils head tail
#' @importFrom car powerTransform


###SUPPORT

globalVariables(c("train_loss", "val_loss", "sequential_kld", "upside_probability"))

nn_mish <- nn_module(
  "nn_mish",
  initialize = function() {self$softplus <- nn_softplus()},
  forward = function(x) {x * torch_tanh(self$softplus(x))})

nn_snake <- nn_module(
  "nn_snake",
  initialize = function(a = 0.5) {self$a <- nn_buffer(a)},
  forward = function(x) {x + torch_square(torch_sin(self$a*x)/self$a)})

nn_bent <- nn_module(
  "nn_bent",
  initialize = function() {},
  forward = function(x) {(torch_sqrt(x^2 + 1) - 1)/2 + x})

nn_swish <- nn_module(
  "nn_swish",
  initialize = function(beta = 1) {self$beta <- nn_buffer(beta)},
  forward = function(x) {x * torch_sigmoid(self$beta * x)})

nn_activ <- nn_module(
  "nn_activ",
  initialize = function(act, dim = 2, alpha = 1, lambda = 0.5, min_val = -1, max_val = 1, a = 0.5)
  {
    if(act == "linear"){self$activ <- nn_identity()}
    if(act == "mish"){self$activ <- nn_mish()}
    if(act == "leaky_relu"){self$activ <- nn_leaky_relu()}
    if(act == "celu"){self$activ <- nn_celu(alpha)}
    if(act == "elu"){self$activ <- nn_elu(alpha)}
    if(act == "gelu"){self$activ <- nn_gelu()}
    if(act == "selu"){self$activ <- nn_selu()}
    if(act == "softplus"){self$activ <- nn_softplus()}
    if(act == "bent"){self$activ <- nn_bent()}
    if(act == "snake"){self$activ <- nn_snake(a)}
    if(act == "softmax"){self$activ <- nn_softmax(dim)}
    if(act == "softmin"){self$activ <- nn_softmin(dim)}
    if(act == "softsign"){self$activ <- nn_softsign()}
    if(act == "sigmoid"){self$activ <- nn_sigmoid()}
    if(act == "tanh"){self$activ <- nn_tanh()}
    if(act == "tanhshrink"){self$activ <- nn_tanhshrink()}
    if(act == "hardsigmoid"){self$activ <- nn_hardsigmoid()}
    if(act == "swish"){self$activ <- nn_swish()}
    if(act == "hardtanh"){self$activ <- nn_hardtanh(min_val, max_val)}
    if(act == "hardshrink"){self$activ <- nn_hardshrink(lambda)}
  },
  forward = function(x)
  {
    x <- self$activ(x)
  })


######

nn_time2vec <- nn_module(
  "nn_time2vec",
  initialize = function(seq_len, k_embed, dev)
  {
    self$trend <- nn_linear(in_features = seq_len, out_features = seq_len, bias = TRUE)$to(device = dev)
    pnames <- paste0("periodic", 1:(k_embed - 1))
    map(pnames, ~ {self[[.x]] <- nn_linear(in_features = seq_len, out_features = seq_len, bias = TRUE)$to(device = dev)})
    self$pnames <- nn_buffer(pnames)
    self$dev <- nn_buffer(dev)
  },
  forward = function(x)
  {
    x <- torch_transpose(x, 3, 2)$to(device = self$dev)
    trend <- self$trend(x)
    pnames <- self$pnames
    periodic <- map(pnames, ~ torch_cos(self[[.x]](x)))
    components <- prepend(periodic, trend)
    components <- map(components, ~ torch_transpose(.x, 3, 2))
    components <- map(components, ~ .x$unsqueeze(4))
    result <- torch_cat(components, dim = 4)
  })

nn_attention <- nn_module(
  "nn_attention",
  initialize = function(k_embed, r_proj, dev)
  {
    #self$linear_Q <- nn_linear(k_embed, r_proj)$to(device = dev)
    self$linear_K <- nn_linear(k_embed, r_proj)$to(device = dev)
    self$linear_V <- nn_linear(k_embed, r_proj)$to(device = dev)
    self$softmax <- nn_softmax(dim = -1)
  },
  forward = function(x)
  {
    #Q <- self$linear_K(x)
    K <- self$linear_K(x) #(n, seq, feat, k)
    V <- self$linear_V(x)

    attn <- torch_matmul(torch_transpose(self$softmax(K), 3, 2), torch_transpose(torch_transpose(V, 3, 2), 4, 3))
    return(torch_transpose(attn, 4, 2))
  })

nn_multi_attention <- nn_module(
  "nn_multi_attention",
  initialize = function(n_heads, k_embed, r_proj, dev)
  {
    head_names <- paste0("att_head_", 1:n_heads)
    map(head_names, ~ {self[[.x]] <- nn_attention(k_embed, r_proj, dev)})
    self$head_names <- nn_buffer(head_names)
  },
  forward = function(x)
  {
    n_heads <- length(self$head_names)
    head_list <- vector(mode = "list", length = n_heads)
    head_list <- map(self$head_names, ~ {head_list[[.x]] <- self[[.x]](x)})
    multi_attn <- torch_cat(head_list, dim = 2)
    return(multi_attn)
  })

nn_variational_parameter <- nn_module(
  "nn_variational_parameter",
  initialize = function(target_len, seq_len, k_embed, r_proj, n_heads, activ, dev)
  {
    self$embedding <- nn_time2vec(seq_len, k_embed, dev)
    self$multi_attention <- nn_multi_attention(n_heads, k_embed, r_proj, dev)
    self$target <- nn_linear(seq_len * n_heads, target_len)$to(device = dev)
    self$focus <- nn_linear(seq_len, 1)$to(device = dev)
    self$activ <- nn_activ(act = activ)
    self$norm <- nn_batch_norm2d(seq_len * n_heads)
  },

  forward = function(x)
  {
    par <- self$embedding(x)
    par <- self$multi_attention(par)
    par <- self$target(torch_transpose(par, 4, 2))
    par <- self$activ(par)
    par <- torch_transpose(torch_transpose(par, 4, 2), 4, 3)
    par <- self$focus(par)
    par <- self$activ(par)
    par <- torch_squeeze(par, 4)

    return(par)
  })


nn_variational_model <- nn_module(
  "nn_variational_model",
  initialize = function(target_len, seq_len, k_embed, r_proj, n_heads, n_bases, activ, dev)
  {
    mean_names <- paste0("mean", 1:n_bases)
    map(mean_names, ~ {self[[.x]] <- nn_variational_parameter(target_len, seq_len, k_embed, r_proj, n_heads, activ, dev)})
    scale_names <- paste0("scale", 1:n_bases)
    map(scale_names, ~ {self[[.x]] <- nn_variational_parameter(target_len, seq_len, k_embed, r_proj, n_heads, activ, dev)})
    self$n_bases <- nn_buffer(n_bases)
    self$mean_names <- nn_buffer(mean_names)
    self$scale_names <- nn_buffer(scale_names)
    self$dev <- nn_buffer(dev)
  },
  forward = function(x)
  {
    n_bases <- self$n_bases
    mean_names <- self$mean_names
    scale_names <- self$scale_names

    mean_list <- map(mean_names, ~self[[.x]](x))
    scale_list <- map(scale_names, ~ torch_square(self[[.x]](x)))

    target_dim <- dim(scale_list[[1]])
    latent_list <- map2(mean_list, scale_list, ~ .x + torch_sqrt(.y) * torch_rand(target_dim, device = self$dev))
    latent <- Reduce("+", latent_list)

    outcome <- list(latent = latent, latent_list = latent_list, mean_list = mean_list, scale_list = scale_list)
    return(outcome)
  })



####
crps_loss <- function(actual, latent, latent_list, mean_list, scale_list, dev)
{
  ###CONTINUOUS RANKED PROBABILITY SCORE (CRPS)
  latent_cdf <- Reduce("+", pmap(list(latent_list, mean_list, scale_list), ~ pnorm(as_array(..1$cpu()), mean = as_array(..2$cpu()), sd = as_array(..3$cpu()))))
  error <- as_array(latent$cpu()) - as_array(actual$cpu())
  heaviside_step <- error >= 0
  loss <- mean((latent_cdf - heaviside_step)^2)###MEAN INSTEAD OF SUM
  loss <- torch_tensor(loss, dtype = torch_float64(), requires_grad = TRUE, device = dev)

  return(loss)
}


elbo_loss <- function(actual, latent, latent_list, mean_list, scale_list, dev)
{
  ###EVIDENCE LOWER BOUND (ELBO)
  recon <- nnf_l1_loss(input = latent, target = actual, reduction = "none")

  latent_pdf <- Reduce("+", pmap(list(latent_list, mean_list, scale_list), ~ dnorm(as_array(..1$cpu()), mean = as_array(..2$cpu()), sd = as_array(..3$cpu()), log = FALSE)))
  log_latent_pdf <- Reduce("+", pmap(list(latent_list, mean_list, scale_list), ~ dnorm(as_array(..1$cpu()), mean = as_array(..2$cpu()), sd = as_array(..3$cpu()), log = TRUE)))
  log_actual_pdf <- Reduce("+", pmap(list(mean_list, scale_list), ~ dnorm(as_array(actual$cpu()), mean = as_array(..1$cpu()), sd = as_array(..2$cpu()), log = TRUE)))

  elbo <- abs(log_actual_pdf * latent_pdf - log_latent_pdf * latent_pdf)
  elbo <- mean(elbo[is.finite(elbo)])
  recon <- as_array(recon$cpu())
  recon <- mean(recon[is.finite(recon)])
  loss <- torch_tensor(recon + elbo, dtype = torch_float64(), requires_grad = TRUE, device = dev)

  return(loss)
}


###PREDICTION
pred_fun <- function(model, new_data, type = "sample", quant = 0.5, seed = as.numeric(Sys.time()), dev, sample_n)
{
  if(!("torch_tensor" %in% class(new_data))){new_data <- torch_tensor(as.array(new_data), device = dev)}
  if(type=="sample"){pred <- as_array(model(new_data)$latent$cpu())}

  if(type == "quant" | type == "mean" | type == "mode")
  {
    pred <- abind(replicate(sample_n, as_array(model(new_data)$latent$cpu()), simplify = FALSE), along=-1)
    if(type == "quant"){pred <- apply(pred, c(2, 3, 4), quantile, probs = quant, na.rm = TRUE)}
    if(type == "mean"){pred <- apply(pred, c(2, 3, 4), mean, na.rm = TRUE)}
    if(type == "mode"){pred <- apply(pred, c(2, 3, 4), function(x) modeest::mlv1(x, method = "parzen"))}
  }

  return(pred)
}

####
smart_split <- function(array, along)
{
  array_split <- split(array, along = along)
  dim_checkout <- dim(array)
  dim_preserve <- dim_checkout[- along]
  array_split <- map(array_split, ~ {dim(.x) <- dim_preserve; return(.x)})
  return(array_split)
}


####
training_function <- function(model, x_train, y_train, x_val, y_val, loss_metric, optim, lr, epochs, patience, verbose, batch_size, dev)
{
  if(optim == "adadelta"){optimizer <- optim_adadelta(model$parameters, lr = lr, rho = 0.9, eps = 1e-06, weight_decay = 0)}
  if(optim == "adagrad"){optimizer <- optim_adagrad(model$parameters, lr = lr, lr_decay = 0, weight_decay = 0, initial_accumulator_value = 0, eps = 1e-10)}
  if(optim == "rmsprop"){optimizer <- optim_rmsprop(model$parameters, lr = lr, alpha = 0.99, eps = 1e-08, weight_decay = 0, momentum = 0, centered = FALSE)}
  if(optim == "rprop"){optimizer <- optim_rprop(model$parameters, lr = lr, etas = c(0.5, 1.2), step_sizes = c(1e-06, 50))}
  if(optim == "sgd"){optimizer <- optim_sgd(model$parameters, lr = lr, momentum = 0, dampening = 0, weight_decay = 0, nesterov = FALSE)}
  if(optim == "asgd"){optimizer <- optim_asgd(model$parameters, lr = lr, lambda = 1e-04, alpha = 0.75, t0 = 1e+06, weight_decay = 0)}
  if(optim == "adam"){optimizer <- optim_adam(model$parameters, lr = lr, betas = c(0.9, 0.999), eps = 1e-08, weight_decay = 0, amsgrad = FALSE)}

  train_history <- vector(mode="numeric", length = epochs)
  val_history <- vector(mode="numeric", length = epochs)
  dynamic_overfit <- vector(mode="numeric", length = epochs)

  x_train <- torch_tensor(x_train, dtype = torch_float32(), device = dev)
  y_train <- torch_tensor(y_train, dtype = torch_float32(), device = dev)
  x_val <- torch_tensor(x_val, dtype = torch_float32(), device = dev)
  y_val <- torch_tensor(y_val, dtype = torch_float32(), device = dev)

  n_train <- nrow(x_train)
  n_val <- nrow(x_val)

  if(batch_size > n_train | batch_size > n_val){batch_size <- min(c(n_train, n_val)); message("setting max batch size to", batch_size,"\n")}

  train_batches <- ceiling(n_train/batch_size)
  train_batch_index <- c(base::rep(1:ifelse(n_train%%batch_size==0, train_batches, train_batches-1), each = batch_size), base::rep(train_batches, each = n_train%%batch_size))

  val_batches <- ceiling(n_val/batch_size)
  val_batch_index <- c(base::rep(1:ifelse(n_val%%batch_size==0, val_batches, val_batches-1), each = batch_size), base::rep(val_batches, each = n_val%%batch_size))
  val_batch_history	<- vector(mode="numeric", length = val_batches)

  parameters <- list()

  for(t in 1:epochs)
  {
    train_batch_history <- vector(mode="numeric", length = train_batches)

    for(b in 1:train_batches)
    {
      index <- b == train_batch_index
      train_results <- model(x_train[index,,])
      if(loss_metric == "elbo"){train_loss <- elbo_loss(actual = y_train[index,,], train_results$latent, train_results$latent_list, train_results$mean_list, train_results$scale_list, dev)}
      if(loss_metric == "crps"){train_loss <- crps_loss(actual = y_train[index,,], train_results$latent, train_results$latent_list, train_results$mean_list, train_results$scale_list, dev)}
      train_batch_history[b] <- train_loss$item()

      train_loss$backward()
      optimizer$step()
      optimizer$zero_grad()
    }

    train_history[t] <- mean(train_batch_history)

    val_batch_history	<- vector(mode="numeric", length = val_batches)

    for(b in 1:val_batches)
    {
      index <- b == val_batch_index
      val_results <- model(x_val[index,,])
      if(loss_metric == "elbo"){val_loss <- elbo_loss(actual = y_val[index,,], val_results$latent, val_results$latent_list, val_results$mean_list, val_results$scale_list, dev)}
      if(loss_metric == "crps"){val_loss <- crps_loss(actual = y_val[index,,], val_results$latent, val_results$latent_list, val_results$mean_list, val_results$scale_list, dev)}
      val_batch_history[b] <- val_loss$item()
    }

    val_history[t] <- mean(val_batch_history)

    if(verbose == TRUE){if (t %% floor(epochs/10) == 0 | epochs < 10) {message("epoch: ", t, "   Train loss: ", train_history[t], "   Val loss: ", val_history[t], "\n")}}

    dynamic_overfit[t] <- abs(val_history[t] - train_history[t])/abs(val_history[1] - train_history[1])
    dyn_ovft_horizon <- c(0, diff(dynamic_overfit[1:t]))
    val_hist_horizon <- c(0, diff(val_history[1:t]))

    if(t >= patience){
      lm_mod1 <- lm(h ~ t, data.frame(t=1:t, h=dyn_ovft_horizon))
      lm_mod2 <- lm(h ~ t, data.frame(t=1:t, h=val_hist_horizon))

      rolling_window <- max(c(patience - t + 1, 1))
      avg_dyn_ovft_deriv <- mean(tail(dyn_ovft_horizon, rolling_window), na.rm = TRUE)
      avg_val_hist_deriv <- mean(tail(val_hist_horizon, rolling_window), na.rm = TRUE)
    }
    if(t >= patience && avg_dyn_ovft_deriv > 0 && lm_mod1$coefficients[2] > 0 && avg_val_hist_deriv > 0 && lm_mod2$coefficients[2] > 0){if(verbose == TRUE){message("early stop at epoch: ", t, "   Train loss: ", train_loss$item(), "   Val loss: ", val_loss$item(), "\n")}; break}
  }

  outcome <- list(model = model, train_history = train_history[1:t], val_history = val_history[1:t])

  return(outcome)
}

###
eval_metrics <- function(actual, predicted)
{
  actual <- unlist(actual)
  predicted <- unlist(predicted)
  if(length(actual) != length(predicted)){stop("different lengths")}

  index <- is.finite(actual) & is.finite(predicted)
  actual <- actual[index]
  predicted <- predicted[index]

  rmse <- sqrt(mean((actual - predicted)^2))
  mae <- mean(abs(actual - predicted))
  mdae <- median(abs(actual - predicted))
  mpe <- mean((actual - predicted)/actual)
  mape <- mean(abs(actual - predicted)/abs(actual))
  smape <- mean(abs(actual - predicted)/mean(c(abs(actual), abs(predicted))))
  rrse <- sqrt(sum((actual - predicted)^2))/sqrt(sum((actual - mean(actual))^2))
  rae <- sum(abs(actual - predicted))/sum(abs(actual - mean(actual)))

  metrics <- round(c(rmse = rmse, mae = mae, mdae = mdae, mpe = mpe, mape = mape, smape = smape, rrse = rrse, rae = rae), 4)
  return(metrics)
}

###

pred_statistics <- function(list_of_predictions, reference_points, future, target, ecdf_by_time)
{
  iqr_to_range <- round(map_dbl(list_of_predictions, ~ mean((.x[,"q75"] - .x[,"q25"])/(.x[,"max"] - .x[,"min"]))), 3)
  dynamic_iqr_ratio <- round(map_dbl(list_of_predictions, ~ (.x[future,"q75"] - .x[future,"q25"])/(.x[1,"q75"] - .x[1,"q25"])), 3)
  upside_prob <- unlist(map2(reference_points, ecdf_by_time, ~ mean(mapply(function(f) (1 - ..2[[f]](..1)), f = 1:future))))
  pred_stats <- round(rbind(iqr_to_range, dynamic_iqr_ratio, upside_prob), 4)
  rownames(pred_stats) <- c("iqr_to_range", "dynamic_iqr_ratio", "upside_prob")
  colnames(pred_stats) <- target

  return(pred_stats)
}


###
reframed_differentiation <- function(reframed, deriv)
{
  ###SUPPORT
  recursive_diff <- function(vector, deriv)
  {
    head_value <- vector("numeric", deriv)
    tail_value <- vector("numeric", deriv)
    if(deriv==0){vector}
    if(deriv > 0){for(i in 1:deriv){head_value[i] <- head(vector, 1); tail_value[i] <- tail(vector, 1); vector <- diff(vector)}}
    outcome <- list(vector, head_value, tail_value)
    return(outcome)
  }
  ###MAIN
  slice_list <- split(reframed, along = 3, drop = TRUE)
  if(nrow(reframed)>1)
  {
    wip_list <- map(slice_list, ~ apply(.x, 1, function(x) recursive_diff(x, deriv)))
    rebind_list <- map(map_depth(wip_list, 2, ~.x[[1]]), ~ Reduce(rbind, .x))
    difframed <- abind(rebind_list, along = 3)
    heads <- map_depth(wip_list, 2, ~.x[[2]])
    tails <- map_depth(wip_list, 2, ~.x[[3]])
  }

  if(nrow(reframed)==1)
  {
    wip_list <- map(slice_list, ~ recursive_diff(.x, deriv))
    rebind_list <- map(wip_list, ~ array(.x[[1]], dim = c(1, length(.x[[1]]), 1)))
    difframed <- abind(rebind_list, along = 3)
    heads <- map(wip_list, ~.x[[2]])
    tails <- map(wip_list, ~.x[[3]])
  }

  outcome <- list(difframed = difframed, heads = heads, tails = tails)
  return(outcome)
}

###
reframed_integration <- function(difframed, heads, deriv)
{
  ###SUPPORT
  invdiff <- function(vector, heads){for(d in length(heads):1)
  {vector <- cumsum(c(heads[d], vector))};return(vector)}

  slice_list <- split(difframed, along = 3, drop = TRUE)
  ###MAIN
  if(nrow(difframed)>1)
  {
    wip_list <- map2(slice_list, heads, ~ t(mapply(function(h) invdiff(.x[h,], .y[[h]]), h = 1:length(.y))))
    integrated <- abind(wip_list, along = 3)
  }

  if(nrow(difframed)==1)
  {
    wip_list <- map2(slice_list, heads, ~ invdiff(.x, .y))
    wip_list <- map2(slice_list, heads, ~ abind(aperm(abind(invdiff(.x, .y), along = -1), c(2, 1)), along = -1))
    integrated <- abind(wip_list, along = 3)
  }

  integrated <- integrated[,- c(1:deriv),,drop=FALSE]
  return(integrated)
}

###
reframe<-function(data, length)
{
  slice_list <- split(data, along=2)
  reframed <- abind(map(slice_list, ~ t(apply(embed(.x, dimension=length), 1, rev))), along=3)
  return(reframed)
}

###
ts_graph <- function(x_hist, y_hist, x_forcat, y_forcat, lower = NULL, upper = NULL, line_size = 1.3, label_size = 11,
                     forcat_band = "darkorange", forcat_line = "darkorange", hist_line = "gray43",
                     label_x = "Horizon", label_y= "Forecasted Var", dbreak = NULL, date_format = "%b-%d-%Y")
{

  all_data <- data.frame("x_all" = c(x_hist, x_forcat), "y_all" = c(y_hist, y_forcat))
  forcat_data <- data.frame("x_forcat" = x_forcat, "y_forcat" = y_forcat)

  if(!is.null(lower) & !is.null(upper)){forcat_data$lower <- lower; forcat_data$upper <- upper}

  plot <- ggplot()+geom_line(data = all_data, aes_string(x = "x_all", y = "y_all"), color = hist_line, size = line_size)
  if(!is.null(lower) & !is.null(upper)){plot <- plot + geom_ribbon(data = forcat_data, aes_string(x = "x_forcat", ymin = "lower", ymax = "upper"), alpha = 0.3, fill = forcat_band)}
  plot <- plot + geom_line(data = forcat_data, aes_string(x = "x_forcat", y = "y_forcat"), color = forcat_line, size = line_size)
  if(!is.null(dbreak)){plot <- plot + scale_x_date(name = paste0("\n", label_x), date_breaks = dbreak, date_labels = date_format)}
  if(is.null(dbreak)){plot <- plot + xlab(label_x)}
  plot <- plot + scale_y_continuous(name = paste0(label_y, "\n"), labels = number)
  plot <- plot + ylab(label_y)  + theme_bw()
  plot <- plot + theme(axis.text=element_text(size=label_size), axis.title=element_text(size=label_size + 2))

  return(plot)
}


###
yjt_fun <- function(x)
{
  yjt <- vector(mode = "numeric", length = length(x))
  scaled <- scale(x)
  x <- as.vector(scaled)
  lambda <- powerTransform(lm(x ~ 1, data.frame(x)), family="yjPower")$lambda

  predict_yjt <- function(x)
  {
  yjt <- vector(mode = "numeric", length = length(x))
  dim(yjt) <- dim(x)

  for(i in 1:length(x))
  {
    if(x[i] >= 0 & lambda != 0){yjt[i] <- ((x[i]+1)^lambda - 1)/lambda}
    if(x[i] >= 0 & lambda == 0){yjt[i] <- log(x[i]+1)}
    if(x[i] < 0 & lambda != 2){yjt[i] <- -((-x[i]+1)^(2 - lambda) - 1)/(2 - lambda)}
    if(x[i] < 0 & lambda == 2){yjt[i] <- -log(-x[i]+1)}
  }
  return(yjt)
  }

  out <- list(lambda = lambda, scaled = scaled, predict_yjt = predict_yjt)
  return(out)
}

###
inv_yjt <- function(x, lambda, scaled)
{
  inv_yjt <- vector(mode = "numeric", length = length(x))
  dim(inv_yjt) <- dim(x)

  for(i in 1:length(x))
  {
    if(x[i] >= 0 & lambda != 0){inv_yjt[i] <- exp(log(x[i] * lambda + 1)/lambda) - 1}
    if(x[i] >= 0 & lambda == 0){inv_yjt[i] <- exp(x[i]) - 1}
    if(x[i] < 0 & lambda != 2){inv_yjt[i] <- 1 - exp(log(1 - x[i]*(2 - lambda))/(2 - lambda))}
    if(x[i] < 0 & lambda == 2){inv_yjt[i] <- 1 - exp(-x[i])}
  }

  rescaled <- inv_yjt * attr(scaled, 'scaled:scale') + attr(scaled, 'scaled:center')

  return(rescaled)
}

###
sequential_kld <- function(m)
{
  matrix <- as.matrix(m)
  n <- nrow(matrix)
  if(n == 1){return(NA)}
  dens <- apply(matrix, 1, function(x) tryCatch(density(x[is.finite(x)], from = min(matrix[is.finite(matrix)]), to = max(matrix[is.finite(matrix)])), error = function(e) NA))
  backward <- dens[-n]
  forward <- dens[-1]

  finite_index <- map2(forward, backward, ~ is.finite(log(.x$y/.y$y)) & is.finite(.x$y))
  seq_kld <- pmap_dbl(list(forward, backward, finite_index), ~ sum(..1$y[..3] * log(..1$y/..2$y)[..3]))
  avg_seq_kld <- round(mean(seq_kld), 3)

  ratios <- log(dens[[n]]$y/dens[[1]]$y)
  finite_index <- is.finite(ratios)

  end_to_end_kld <- dens[[n]]$y * log(dens[[n]]$y/dens[[1]]$y)
  end_to_end_kld <- tryCatch(round(sum(end_to_end_kld[finite_index]), 3), error = function(e) NA)
  kld_stats <- rbind(avg_seq_kld, end_to_end_kld)

  return(kld_stats)
}

###
upside_probability <- function(m)
{
  matrix <- as.matrix(m)
  n <- nrow(matrix)
  if(n == 1){return(NA)}
  growths <- matrix[-1,]/matrix[-n,] - 1
  dens <- apply(growths, 1, function(x) tryCatch(density(x[is.finite(x)], from = min(x[is.finite(x)]), to = max(x[is.finite(x)])), error = function(e) NA))
  not_na <- !is.na(dens)
  avg_upp <- tryCatch(round(mean(map_dbl(dens[not_na], ~ sum(.x$y[.x$x>0])/sum(.x$y))), 3), error = function(e) NA)
  end_growth <- matrix[n,]/matrix[1,] - 1
  end_to_end_dens <- tryCatch(density(end_growth[is.finite(end_growth)], from = min(end_growth[is.finite(end_growth)]), to = max(end_growth[is.finite(end_growth)])), error = function(e) NA)
  if(class(end_to_end_dens) == "density"){last_to_first_upp <- round(sum(end_to_end_dens$y[end_to_end_dens$x>0])/sum(end_to_end_dens$y), 3)} else {last_to_first_upp <- NA}
  upp_stats <- rbind(avg_upp, last_to_first_upp)
  return(upp_stats)
}
