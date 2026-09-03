# Mixed-outcome multitask learning by accelerated proximal gradient
#
# Allowed task types:
#   "regression"
#   "Huber regression"
#   "adaptive Huber"
#   "logistic regression"
#   "AFT exponential"
#
# The fitted objective is
#
#   sum_k w_k L_k(beta_k)
#     + lambda1 * sum_j ||beta[j, ]||_2
#     + lambda2 * sum_g sum_{k in G_g} ||beta_k - B_g||_2^2,
#
# where B_g is the mean coefficient vector within task group g.


## -----------------------------------------------------------------
## Proximal operator for the row-wise group L2,1 penalty
## -----------------------------------------------------------------

.group_penalty <- function(z, lambda, gamma = 1) {
  
  z <- as.matrix(z)
  
  if (length(lambda) != 1L || !is.finite(lambda) || lambda < 0) {
    stop("lambda must be one nonnegative finite number.")
  }
  
  if (length(gamma) != 1L || !is.finite(gamma) || gamma <= 0) {
    stop("gamma must be one positive finite number.")
  }
  
  row_norm <- sqrt(rowSums(z^2))
  
  shrink <- pmax(
    0,
    1 - lambda / (gamma * pmax(row_norm, 1e-12))
  )
  
  sweep(
    z,
    MARGIN = 1,
    STATS = shrink,
    FUN = "*"
  )
}


## -----------------------------------------------------------------
## Within-task-group similarity penalty
## -----------------------------------------------------------------

.similarity_penalty_value <- function(beta, task_groups) {
  
  beta <- as.matrix(beta)
  value <- 0
  
  for (g in seq_along(task_groups)) {
    
    idx <- task_groups[[g]]
    
    if (length(idx) <= 1L) {
      next
    }
    
    beta_g <- beta[, idx, drop = FALSE]
    B_g <- rowMeans(beta_g)
    
    difference <- sweep(
      beta_g,
      MARGIN = 1,
      STATS = B_g,
      FUN = "-"
    )
    
    value <- value + sum(difference^2)
  }
  
  value
}


.similarity_gradient <- function(beta, task_groups) {
  
  beta <- as.matrix(beta)
  
  gradient <- matrix(
    0,
    nrow = nrow(beta),
    ncol = ncol(beta),
    dimnames = dimnames(beta)
  )
  
  for (g in seq_along(task_groups)) {
    
    idx <- task_groups[[g]]
    
    if (length(idx) <= 1L) {
      next
    }
    
    beta_g <- beta[, idx, drop = FALSE]
    B_g <- rowMeans(beta_g)
    
    difference <- sweep(
      beta_g,
      MARGIN = 1,
      STATS = B_g,
      FUN = "-"
    )
    
    # The factor 2 corresponds to a penalty defined as sum(difference^2).
    gradient[, idx] <-
      gradient[, idx, drop = FALSE] + 2 * difference
  }
  
  gradient
}


## -----------------------------------------------------------------
## Continuous-outcome loss
## x must already be processed/standardized before this function.
## -----------------------------------------------------------------

.cont_loss <- function(x, y, beta, tau = NULL, method = "regression") {
  
  x <- as.matrix(x)
  y <- as.numeric(y)
  beta <- as.numeric(beta)
  
  n <- nrow(x)
  
  if (length(y) != n || length(beta) != ncol(x)) {
    stop("Incompatible x, y, and beta dimensions in .cont_loss().")
  }
  
  method_key <- tolower(trimws(method))
  u_raw <- y - as.vector(x %*% beta)
  
  if (method_key == "regression") {
    
    u <- u_raw
    score <- -as.vector(crossprod(x, u)) / n
    loss <- sum(u_raw^2) / (2 * n)
    
  } else if (method_key %in% c("huber regression", "adaptive huber")) {
    
    if (length(tau) != 1L || !is.finite(tau) || tau <= 0) {
      stop("tau must be one positive finite number for a Huber loss.")
    }
    
    abs_u <- abs(u_raw)
    u <- pmax(pmin(u_raw, tau), -tau)
    
    score <- -as.vector(crossprod(x, u)) / n
    
    loss <- sum(
      ifelse(
        abs_u <= tau,
        0.5 * u_raw^2,
        tau * abs_u - 0.5 * tau^2
      )
    ) / n
    
  } else {
    stop(paste0("Unknown continuous-loss method: ", method))
  }
  
  list(
    score = score,
    loss = as.numeric(loss),
    tau = tau,
    residual = u_raw,
    robust_score_residual = u
  )
}


## -----------------------------------------------------------------
## Logistic negative log-likelihood
## -----------------------------------------------------------------

.binary_loss <- function(x, y, beta) {
  
  x <- as.matrix(x)
  y <- as.numeric(y)
  beta <- as.numeric(beta)
  
  n <- nrow(x)
  
  if (length(y) != n || length(beta) != ncol(x)) {
    stop("Incompatible x, y, and beta dimensions in .binary_loss().")
  }
  
  if (!all(y %in% c(0, 1))) {
    stop("For logistic regression, y must be coded as 0/1.")
  }
  
  eta <- as.vector(x %*% beta)
  mu <- plogis(eta)
  u <- y - mu
  
  score <- -as.vector(crossprod(x, u)) / n
  
  # Stable evaluation of log(1 + exp(eta)) - y * eta.
  loss <- sum(
    pmax(eta, 0) + log1p(exp(-abs(eta))) - y * eta
  ) / n
  
  list(
    score = score,
    loss = as.numeric(loss),
    residual = u,
    fitted_probability = mu
  )
}


## -----------------------------------------------------------------
## Exponential AFT negative log-likelihood
##
## L(beta) = mean[y * exp(-x beta) + delta * x beta]
## grad L  = X' [delta - y * exp(-x beta)] / n
## -----------------------------------------------------------------

.aft_exp_loss <- function(x, y, delta, beta) {
  
  x <- as.matrix(x)
  y <- as.numeric(y)
  delta <- as.numeric(delta)
  beta <- as.numeric(beta)
  
  n <- nrow(x)
  
  if (length(y) != n || length(delta) != n ||
      length(beta) != ncol(x)) {
    stop("Incompatible x, y, delta, and beta dimensions in .aft_exp_loss().")
  }
  
  if (any(y < 0)) {
    stop("For AFT exponential tasks, all observed times must be nonnegative.")
  }
  
  if (!all(delta %in% c(0, 1))) {
    stop("For AFT exponential tasks, delta must be coded as 0/1.")
  }
  
  eta <- as.vector(x %*% beta)
  
  # Capping only protects the exponential calculation from overflow.
  # Under ordinary fitted values this cap is inactive.
  hazard_rate <- exp(pmin(-eta, 700))
  u <- delta - hazard_rate * y
  
  score <- as.vector(crossprod(x, u)) / n
  loss <- sum(hazard_rate * y + delta * eta) / n
  
  list(
    score = score,
    loss = as.numeric(loss),
    residual = u,
    hazard_rate = hazard_rate,
    linear_predictor = eta
  )
}


## -----------------------------------------------------------------
## Single task dispatcher
## -----------------------------------------------------------------

.task_loss <- function(task, beta) {
  
  type_key <- tolower(trimws(task$type))
  
  if (type_key == "regression") {
    
    out <- .cont_loss(
      x = task$x,
      y = task$y,
      beta = beta,
      method = "regression"
    )
    
  } else if (type_key == "huber regression") {
    
    out <- .cont_loss(
      x = task$x,
      y = task$y,
      beta = beta,
      tau = task$tau_effective,
      method = "Huber regression"
    )
    
  } else if (type_key == "adaptive huber") {
    
    out <- .cont_loss(
      x = task$x,
      y = task$y,
      beta = beta,
      tau = task$tau_effective,
      method = "adaptive Huber"
    )
    
  } else if (type_key == "logistic regression") {
    
    out <- .binary_loss(
      x = task$x,
      y = task$y,
      beta = beta
    )
    
  } else if (type_key == "aft exponential") {
    
    out <- .aft_exp_loss(
      x = task$x,
      y = task$y,
      delta = task$delta,
      beta = beta
    )
    
  } else {
    
    stop(
      paste0(
        "Unknown task type: ", task$type, ". Allowed types are: ",
        "regression, Huber regression, adaptive Huber, ",
        "logistic regression, and AFT exponential."
      )
    )
  }
  
  out
}


## -----------------------------------------------------------------
## Main algorithm
## -----------------------------------------------------------------

.Main_Algo <- function(
    tasks,
    lambda1,
    lambda2 = 0,
    task_groups = NULL,
    beta = 0.1,
    gamma = 1,
    standardize = TRUE,
    adaptive_divisor = 10,
    tol = 1e-5,
    selection_tol = 1e-10,
    max_iter = 500,
    max_backtrack = 30,
    accelerate = TRUE,
    verbose = FALSE
) {
  
  ## ---------------------------------------------------------------
  ## 1. Validate top-level arguments
  ## ---------------------------------------------------------------
  
  K <- length(tasks)
  
  if (K == 0L) {
    stop("tasks cannot be empty.")
  }
  
  if (length(lambda1) != 1L || !is.finite(lambda1) || lambda1 < 0) {
    stop("lambda1 must be one nonnegative finite number.")
  }
  
  if (length(lambda2) != 1L || !is.finite(lambda2) || lambda2 < 0) {
    stop("lambda2 must be one nonnegative finite number.")
  }
  
  if (length(gamma) != 1L || !is.finite(gamma) || gamma <= 0) {
    stop("gamma must be one positive finite number.")
  }
  
  if (length(tol) != 1L || !is.finite(tol) || tol <= 0) {
    stop("tol must be one positive finite number.")
  }
  
  if (length(selection_tol) != 1L ||
      !is.finite(selection_tol) || selection_tol < 0) {
    stop("selection_tol must be one nonnegative finite number.")
  }
  
  if (length(max_iter) != 1L || max_iter < 1 ||
      max_iter != as.integer(max_iter)) {
    stop("max_iter must be a positive integer.")
  }
  
  if (length(max_backtrack) != 1L || max_backtrack < 1 ||
      max_backtrack != as.integer(max_backtrack)) {
    stop("max_backtrack must be a positive integer.")
  }
  
  if (length(adaptive_divisor) != 1L ||
      !is.finite(adaptive_divisor) || adaptive_divisor <= 0) {
    stop("adaptive_divisor must be one positive finite number.")
  }
  
  allowed_types <- c(
    "regression",
    "huber regression",
    "adaptive huber",
    "logistic regression",
    "aft exponential"
  )
  
  
  ## ---------------------------------------------------------------
  ## 2. Validate and preprocess tasks
  ## ---------------------------------------------------------------
  
  tasks_work <- tasks
  
  if (is.null(tasks_work[[1]]$x)) {
    stop("Every task must contain x.")
  }
  
  tasks_work[[1]]$x <- as.matrix(tasks_work[[1]]$x)
  p <- ncol(tasks_work[[1]]$x)
  
  if (is.null(p) || p < 1L) {
    stop("Every task must contain at least one predictor.")
  }
  
  first_feature_names <- colnames(tasks_work[[1]]$x)
  
  if (is.null(first_feature_names)) {
    first_feature_names <- paste0("Feature_", seq_len(p))
  }
  
  n <- integer(K)
  task_weights <- numeric(K)
  x_scales <- matrix(
    1,
    nrow = p,
    ncol = K,
    dimnames = list(first_feature_names, NULL)
  )
  
  canonical_type <- character(K)
  
  for (k in seq_len(K)) {
    
    task <- tasks_work[[k]]
    
    required_fields <- c("x", "y", "type")
    missing_fields <- required_fields[
      !vapply(required_fields, function(z) !is.null(task[[z]]), logical(1))
    ]
    
    if (length(missing_fields) > 0L) {
      stop(
        paste0(
          "Task ", k, " is missing: ",
          paste(missing_fields, collapse = ", "), "."
        )
      )
    }
    
    task$x <- as.matrix(task$x)
    storage.mode(task$x) <- "double"
    task$y <- as.numeric(task$y)
    
    if (ncol(task$x) != p) {
      stop("All tasks must have the same number of predictors.")
    }
    
    feature_names_k <- colnames(task$x)
    
    if (!is.null(feature_names_k) &&
        !identical(feature_names_k, first_feature_names)) {
      stop(
        paste0(
          "Predictor names and order for task ", k,
          " do not match the first task."
        )
      )
    }
    
    colnames(task$x) <- first_feature_names
    
    n[k] <- nrow(task$x)
    
    if (length(task$y) != n[k]) {
      stop(paste0("x and y dimensions do not agree for task ", k, "."))
    }
    
    if (anyNA(task$x) || any(!is.finite(task$x))) {
      stop(paste0("task$x contains missing or non-finite values for task ", k, "."))
    }
    
    if (anyNA(task$y) || any(!is.finite(task$y))) {
      stop(paste0("task$y contains missing or non-finite values for task ", k, "."))
    }
    
    type_key <- tolower(trimws(task$type))
    
    if (length(type_key) != 1L || !type_key %in% allowed_types) {
      stop(
        paste0(
          "Unknown type for task ", k, ": ", task$type,
          ". Allowed types are: regression, Huber regression, ",
          "adaptive Huber, logistic regression, and AFT exponential."
        )
      )
    }
    
    canonical_type[k] <- switch(
      type_key,
      "regression" = "regression",
      "huber regression" = "Huber regression",
      "adaptive huber" = "adaptive Huber",
      "logistic regression" = "logistic regression",
      "aft exponential" = "AFT exponential"
    )
    
    task$type <- canonical_type[k]
    
    task$weight <- if (is.null(task$weight)) 1 else task$weight
    
    if (length(task$weight) != 1L ||
        !is.finite(task$weight) || task$weight <= 0) {
      stop(paste0("Task weight must be positive and finite for task ", k, "."))
    }
    
    task_weights[k] <- task$weight
    
    if (type_key == "logistic regression" &&
        !all(task$y %in% c(0, 1))) {
      stop(paste0("Logistic y must be coded as 0/1 for task ", k, "."))
    }
    
    if (type_key == "aft exponential") {
      
      if (is.null(task$delta)) {
        stop(paste0("delta is required for AFT exponential task ", k, "."))
      }
      
      task$delta <- as.numeric(task$delta)
      
      if (length(task$delta) != n[k] ||
          anyNA(task$delta) ||
          !all(task$delta %in% c(0, 1))) {
        stop(paste0("delta must be a length-n 0/1 vector for task ", k, "."))
      }
      
      if (any(task$y < 0)) {
        stop(paste0("AFT observed times must be nonnegative for task ", k, "."))
      }
    }
    
    if (type_key == "huber regression") {
      
      task$tau <- if (is.null(task$tau)) 1.45 else task$tau
      
      if (length(task$tau) != 1L ||
          !is.finite(task$tau) || task$tau <= 0) {
        stop(paste0("tau must be positive and finite for task ", k, "."))
      }
      
      task$tau_effective <- task$tau
    }
    
    if (type_key == "adaptive huber") {
      
      # Here task$tau is the base multiplier. The effective threshold is
      # calculated once below and then held fixed during optimization.
      task$tau <- if (is.null(task$tau)) 1.45 else task$tau
      
      if (length(task$tau) != 1L ||
          !is.finite(task$tau) || task$tau <= 0) {
        stop(paste0("tau must be positive and finite for task ", k, "."))
      }
    }
    
    if (standardize) {
      
      scale_k <- apply(task$x, 2, stats::sd)
      scale_k[!is.finite(scale_k) | scale_k == 0] <- 1
      
      task$x <- sweep(
        task$x,
        MARGIN = 2,
        STATS = scale_k,
        FUN = "/"
      )
      
      x_scales[, k] <- scale_k
    }
    
    tasks_work[[k]] <- task
  }
  
  
  ## ---------------------------------------------------------------
  ## 3. Initialize beta
  ## ---------------------------------------------------------------
  
  if (length(beta) == 1L) {
    
    beta <- matrix(
      beta,
      nrow = p,
      ncol = K
    )
    
  } else {
    
    beta <- as.matrix(beta)
    
    if (!identical(dim(beta), c(p, K))) {
      stop("Initial beta must be a p by K matrix.")
    }
    
    if (anyNA(beta) || any(!is.finite(beta))) {
      stop("Initial beta must contain only finite values.")
    }
  }
  
  if (anyNA(beta) || any(!is.finite(beta))) {
    stop("Initial beta must contain only finite values.")
  }
  
  rownames(beta) <- first_feature_names
  
  
  ## ---------------------------------------------------------------
  ## 4. Freeze each adaptive-Huber threshold before optimization
  ## ---------------------------------------------------------------
  
  for (k in seq_len(K)) {
    
    if (tolower(tasks_work[[k]]$type) != "adaptive huber") {
      next
    }
    
    initial_residual <- tasks_work[[k]]$y -
      as.vector(tasks_work[[k]]$x %*% beta[, k])
    
    residual_scale <- stats::sd(initial_residual)
    
    if (!is.finite(residual_scale) || residual_scale == 0) {
      residual_scale <- 1
    }
    
    tasks_work[[k]]$tau_effective <-
      tasks_work[[k]]$tau *
      residual_scale *
      sqrt(n[k] / log(max(p, 2))) /
      adaptive_divisor
  }
  
  
  ## ---------------------------------------------------------------
  ## 5. Validate similarity groups
  ## ---------------------------------------------------------------
  
  use_similarity <- lambda2 > 0
  
  if (use_similarity && is.null(task_groups)) {
    stop("task_groups must be supplied when lambda2 > 0.")
  }
  
  if (!is.null(task_groups)) {
    
    if (!is.list(task_groups)) {
      stop("task_groups must be a list of task-index vectors.")
    }
    
    for (g in seq_along(task_groups)) {
      
      idx <- task_groups[[g]]
      
      if (!is.numeric(idx) || length(idx) == 0L ||
          any(!is.finite(idx)) ||
          any(idx != as.integer(idx)) ||
          any(idx < 1L | idx > K) ||
          anyDuplicated(idx) > 0L) {
        stop(
          paste0(
            "Task group ", g,
            " must contain unique integer task indices between 1 and K."
          )
        )
      }
      
      task_groups[[g]] <- as.integer(idx)
    }
    
    all_group_indices <- unlist(task_groups, use.names = FALSE)
    
    if (anyDuplicated(all_group_indices) > 0L) {
      stop(
        "The supplied task groups overlap. Each task should belong to at most one group."
      )
    }
  }
  
  
  ## ---------------------------------------------------------------
  ## 6. Task names
  ## ---------------------------------------------------------------
  
  task_names <- names(tasks)
  
  if (is.null(task_names)) {
    task_names <- paste0("Task_", seq_len(K))
  } else {
    empty_name <- is.na(task_names) | trimws(task_names) == ""
    task_names[empty_name] <- paste0("Task_", which(empty_name))
  }
  
  if (anyDuplicated(task_names) > 0L) {
    stop("Task names must be unique.")
  }
  
  colnames(beta) <- task_names
  colnames(x_scales) <- task_names
  
  
  ## ---------------------------------------------------------------
  ## 7. Loss and gradient evaluator
  ## ---------------------------------------------------------------
  
  evaluate_loss <- function(beta_now) {
    
    gradient <- matrix(
      0,
      nrow = p,
      ncol = K,
      dimnames = dimnames(beta_now)
    )
    
    loss <- numeric(K)
    
    for (k in seq_len(K)) {
      
      out <- .task_loss(
        task = tasks_work[[k]],
        beta = beta_now[, k]
      )
      
      gradient[, k] <- as.numeric(out$score)
      loss[k] <- as.numeric(out$loss)
    }
    
    names(loss) <- task_names
    
    weighted_gradient <- sweep(
      gradient,
      MARGIN = 2,
      STATS = task_weights,
      FUN = "*"
    )
    
    list(
      gradient = weighted_gradient,
      unweighted_gradient = gradient,
      loss = loss,
      weighted_loss = sum(task_weights * loss)
    )
  }
  
  
  group_penalty_value <- function(beta_now) {
    sum(sqrt(rowSums(beta_now^2)))
  }
  
  
  smooth_objective <- function(beta_now) {
    
    out <- evaluate_loss(beta_now)
    value <- out$weighted_loss
    
    if (use_similarity) {
      value <- value + lambda2 *
        .similarity_penalty_value(beta_now, task_groups)
    }
    
    value
  }
  
  
  ## ---------------------------------------------------------------
  ## 8. Accelerated proximal-gradient iterations
  ## ---------------------------------------------------------------
  
  beta_current <- beta
  theta <- beta
  t_old <- 1
  converged <- FALSE
  iter_done <- 0L
  
  history <- data.frame(
    iteration = seq_len(max_iter),
    loss = rep(NA_real_, max_iter),
    group_penalty = rep(NA_real_, max_iter),
    similarity_penalty = rep(NA_real_, max_iter),
    objective = rep(NA_real_, max_iter),
    difference = rep(NA_real_, max_iter),
    gamma = rep(NA_real_, max_iter),
    backtracking_steps = rep(NA_integer_, max_iter)
  )
  
  for (iter in seq_len(max_iter)) {
    
    eval_theta <- evaluate_loss(theta)
    gradient <- eval_theta$gradient
    
    if (use_similarity) {
      gradient <- gradient + lambda2 *
        .similarity_gradient(theta, task_groups)
    }
    
    f_theta <- eval_theta$weighted_loss
    
    if (use_similarity) {
      f_theta <- f_theta + lambda2 *
        .similarity_penalty_value(theta, task_groups)
    }
    
    gamma_try <- gamma
    accepted <- FALSE
    
    for (bt in seq_len(max_backtrack)) {
      
      z <- theta - gradient / gamma_try
      
      beta_new <- .group_penalty(
        z = z,
        lambda = lambda1,
        gamma = gamma_try
      )
      
      f_new <- smooth_objective(beta_new)
      delta_beta <- beta_new - theta
      
      upper_bound <-
        f_theta +
        sum(gradient * delta_beta) +
        gamma_try * sum(delta_beta^2) / 2
      
      if (is.finite(f_new) &&
          f_new <= upper_bound + 1e-12 * max(1, abs(f_theta))) {
        accepted <- TRUE
        break
      }
      
      gamma_try <- gamma_try * 2
    }
    
    if (!accepted) {
      stop(
        "Backtracking failed to find an acceptable step. ",
        "Increase max_backtrack or check the loss-gradient calculations."
      )
    }
    
    gamma <- gamma_try
    
    difference <- max(abs(beta_new - beta_current))
    eval_new <- evaluate_loss(beta_new)
    
    penalty1 <- lambda1 * group_penalty_value(beta_new)
    
    penalty2 <- if (use_similarity) {
      lambda2 * .similarity_penalty_value(beta_new, task_groups)
    } else {
      0
    }
    
    objective <- eval_new$weighted_loss + penalty1 + penalty2
    
    history[iter, c(
      "loss",
      "group_penalty",
      "similarity_penalty",
      "objective",
      "difference",
      "gamma",
      "backtracking_steps"
    )] <- list(
      eval_new$weighted_loss,
      penalty1,
      penalty2,
      objective,
      difference,
      gamma,
      bt
    )
    
    iter_done <- iter
    
    if (verbose) {
      cat(
        "Iteration:", iter,
        "| Objective:", signif(objective, 7),
        "| Difference:", signif(difference, 5),
        "| Gamma:", signif(gamma, 5),
        "| Backtracking:", bt,
        "\n"
      )
    }
    
    if (difference <= tol) {
      beta_current <- beta_new
      converged <- TRUE
      break
    }
    
    if (accelerate) {
      
      t_new <- (1 + sqrt(1 + 4 * t_old^2)) / 2
      
      theta <- beta_new +
        ((t_old - 1) / t_new) * (beta_new - beta_current)
      
      t_old <- t_new
      
    } else {
      theta <- beta_new
    }
    
    beta_current <- beta_new
  }
  
  history <- history[seq_len(iter_done), , drop = FALSE]
  
  
  ## ---------------------------------------------------------------
  ## 9. Final estimates and output summaries
  ## ---------------------------------------------------------------
  
  beta_hat <- beta_current
  rownames(beta_hat) <- first_feature_names
  colnames(beta_hat) <- task_names
  
  # Since x_standardized = x_original / x_scale,
  # beta_original = beta_standardized / x_scale.
  beta_original_scale <- beta_hat / x_scales
  
  selected_variables <- which(
    sqrt(rowSums(beta_hat^2)) > selection_tol
  )
  
  selected_names <- rownames(beta_hat)[selected_variables]
  
  B_hat <- NULL
  
  if (use_similarity) {
    
    B_hat <- vapply(
      task_groups,
      function(idx) rowMeans(beta_hat[, idx, drop = FALSE]),
      numeric(p)
    )
    
    if (is.null(dim(B_hat))) {
      B_hat <- matrix(B_hat, ncol = 1L)
    }
    
    group_names <- names(task_groups)
    
    if (is.null(group_names)) {
      group_names <- paste0("Group_", seq_along(task_groups))
    } else {
      empty_group_name <- is.na(group_names) | trimws(group_names) == ""
      group_names[empty_group_name] <-
        paste0("Group_", which(empty_group_name))
    }
    
    rownames(B_hat) <- first_feature_names
    colnames(B_hat) <- group_names
  }
  
  tau_effective <- vapply(
    tasks_work,
    function(task) {
      if (is.null(task$tau_effective)) NA_real_ else task$tau_effective
    },
    numeric(1)
  )
  
  task_information <- data.frame(
    task = seq_len(K),
    task_name = task_names,
    type = canonical_type,
    n = n,
    weight = task_weights,
    tau_effective = tau_effective,
    stringsAsFactors = FALSE
  )
  
  final_eval <- evaluate_loss(beta_hat)
  
  final_group_penalty <- lambda1 * group_penalty_value(beta_hat)
  
  final_similarity_penalty <- if (use_similarity) {
    lambda2 * .similarity_penalty_value(beta_hat, task_groups)
  } else {
    0
  }
  
  output <- list(
    beta = beta_hat,
    beta_original_scale = beta_original_scale,
    B = B_hat,
    selected_variables = selected_variables,
    selected_names = selected_names,
    task_information = task_information,
    task_weights = stats::setNames(task_weights, task_names),
    task_groups = task_groups,
    lambda1 = lambda1,
    lambda2 = lambda2,
    loss_by_task = final_eval$loss,
    weighted_loss = final_eval$weighted_loss,
    group_penalty = final_group_penalty,
    similarity_penalty = final_similarity_penalty,
    objective = final_eval$weighted_loss +
      final_group_penalty + final_similarity_penalty,
    history = history,
    converged = converged,
    iterations = iter_done,
    final_difference = history$difference[iter_done],
    final_gamma = gamma,
    standardized = standardize,
    x_scales = x_scales,
    call = match.call()
  )
  
  class(output) <- "mixed_mtl_fit"
  output
}




.info_fitting <- function(
    tasks,
    beta,
    import_w = 1,
    alpha = 1,
    tol = 1e-6,
    max_iter = 500,
    selection_tol = 1e-10,
    paired_tasks = FALSE,
    tau_effective = NULL
) {
  
  import_w_missing <- missing(import_w)
  K <- length(tasks)
  selection_beta <- NULL
  
  if (K == 0L) {
    stop("tasks cannot be empty.")
  }
  
  ## A complete mixed_mtl_fit object may be supplied in place of beta.
  ## This automatically uses coefficients on the original predictor scale.
  if (inherits(beta, "mixed_mtl_fit")) {
    fit_object <- beta
    
    if (is.null(fit_object$beta_original_scale)) {
      stop("The fitted object does not contain beta_original_scale.")
    }
    
    if (import_w_missing) {
      import_w <- as.numeric(fit_object$task_weights)
    }
    
    if (is.null(tau_effective) &&
        !is.null(fit_object$task_information$tau_effective)) {
      tau_effective <- fit_object$task_information$tau_effective
    }
    
    selection_beta <- fit_object$beta
    beta <- fit_object$beta_original_scale
  }
  
  beta <- as.matrix(beta)
  
  if (is.null(selection_beta)) {
    selection_beta <- beta
  } else {
    selection_beta <- as.matrix(selection_beta)
  }
  
  p <- nrow(beta)
  
  if (is.null(p) || p < 1L || ncol(beta) != K ||
      !identical(dim(selection_beta), dim(beta))) {
    stop("beta must be a p by K matrix, with one column per task.")
  }
  
  if (anyNA(beta) || any(!is.finite(beta)) ||
      anyNA(selection_beta) || any(!is.finite(selection_beta))) {
    stop("beta must contain only finite values.")
  }
  
  if (length(alpha) != 1L || !is.finite(alpha) || alpha < 0) {
    stop("alpha must be one nonnegative finite number.")
  }
  
  if (length(tol) != 1L || !is.finite(tol) || tol <= 0) {
    stop("tol must be one positive finite number.")
  }
  
  if (length(max_iter) != 1L || max_iter < 1L ||
      max_iter != as.integer(max_iter)) {
    stop("max_iter must be a positive integer.")
  }
  
  if (length(selection_tol) != 1L ||
      !is.finite(selection_tol) || selection_tol < 0) {
    stop("selection_tol must be one nonnegative finite number.")
  }
  
  if (length(paired_tasks) != 1L || is.na(paired_tasks)) {
    stop("paired_tasks must be TRUE or FALSE.")
  }
  
  ## ============================================================
  ## 1. Importance weights
  ## ============================================================
  
  if (length(import_w) == 1) {
    import_w <- rep(import_w, K)
  }
  
  if (length(import_w) != K || anyNA(import_w) ||
      any(!is.finite(import_w)) || any(import_w < 0)) {
    stop("import_w must contain K nonnegative finite weights.")
  }
  
  if (!is.null(tau_effective) && length(tau_effective) != K) {
    stop("tau_effective must be NULL or have one value per task.")
  }
  
  
  ## ============================================================
  ## 2. Validate and canonicalize task inputs
  ## ============================================================
  
  allowed_types <- c(
    "regression",
    "huber regression",
    "adaptive huber",
    "logistic regression",
    "aft exponential"
  )
  
  canonical_type <- c(
    "regression" = "regression",
    "huber regression" = "Huber regression",
    "adaptive huber" = "adaptive Huber",
    "logistic regression" = "logistic regression",
    "aft exponential" = "AFT exponential"
  )
  
  tasks_work <- tasks
  n <- integer(K)
  type <- character(K)
  
  for (k in seq_len(K)) {
    task <- tasks_work[[k]]
    
    if (is.null(task$x) || is.null(task$y) || is.null(task$type)) {
      stop(paste0("Task ", k, " must contain x, y, and type."))
    }
    
    task$x <- as.matrix(task$x)
    storage.mode(task$x) <- "double"
    task$y <- as.numeric(task$y)
    
    if (ncol(task$x) != p) {
      stop(paste0("Task ", k, " does not have ", p, " predictors."))
    }
    
    n[k] <- nrow(task$x)
    
    if (length(task$y) != n[k] || anyNA(task$x) ||
        any(!is.finite(task$x)) || anyNA(task$y) ||
        any(!is.finite(task$y))) {
      stop(paste0("Task ", k, " contains incompatible or non-finite data."))
    }
    
    type_key <- tolower(trimws(task$type))
    
    if (length(type_key) != 1L || !type_key %in% allowed_types) {
      stop(paste0("Unknown task type for task ", k, ": ", task$type, "."))
    }
    
    task$type <- unname(canonical_type[type_key])
    type[k] <- task$type
    
    if (type_key == "logistic regression" &&
        !all(task$y %in% c(0, 1))) {
      stop(paste0("Logistic y must be coded as 0/1 for task ", k, "."))
    }
    
    if (type_key == "aft exponential") {
      if (is.null(task$delta)) {
        stop(paste0("delta is required for AFT task ", k, "."))
      }
      
      task$delta <- as.numeric(task$delta)
      
      if (length(task$delta) != n[k] || anyNA(task$delta) ||
          !all(task$delta %in% c(0, 1)) || any(task$y < 0)) {
        stop(paste0(
          "AFT task ", k,
          " requires nonnegative times and a length-n 0/1 delta vector."
        ))
      }
    }
    
    tasks_work[[k]] <- task
  }
  
  task_names <- names(tasks)
  
  if (is.null(task_names)) {
    task_names <- paste0("Task_", seq_len(K))
  } else {
    empty_name <- is.na(task_names) | trimws(task_names) == ""
    task_names[empty_name] <- paste0("Task_", which(empty_name))
  }
  
  if (anyDuplicated(task_names) > 0L) {
    stop("Task names must be unique.")
  }
  
  
  ## ============================================================
  ## 3. Selected features
  ##
  ## IMPORTANT:
  ## Do NOT use beta[,1] only.
  ## Group sparsity means a variable is selected if it is nonzero
  ## in at least one task.
  ## ============================================================
  
  feature <- which(
    rowSums(abs(selection_beta) > selection_tol) > 0
  )
  
  p1 <- length(feature)
  
  
  if (p1 == 0) {
    
    warning("No variable is selected; penalty value may be too large.")
    
    return(
      list(
        Bayesian_Information = c(
          "Compo_likelihood" = NA,
          "Degree of freedom" = NA,
          "Info criterion" = NA
        ),
        Selected_Features = integer(0),
        Refit_beta = NULL,
        Task_Performance = NULL,
        Reg_Error = NULL,
        Class_Perform = NULL,
        Surv_Perform = NULL,
        Residuals = NULL,
        Score_Residuals = NULL
      )
    )
  }
  
  
  ## ============================================================
  ## 4. Basic task information
  ## ============================================================
  
  if (p1 >= min(n)) {
    
    warning("Number of selected variables >= minimum task sample size.")
    
  }
  
  
  ## ============================================================
  ## 5. Storage
  ## ============================================================
  
  max_n <- max(n)
  
  residuals <- matrix(
    NA_real_,
    nrow = max_n,
    ncol = K
  )
  
  score_residuals <- matrix(
    NA_real_,
    nrow = max_n,
    ncol = K
  )
  
  colnames(residuals) <- task_names
  colnames(score_residuals) <- task_names
  
  
  ## Re-estimated coefficients using selected variables
  
  refit_beta <- matrix(
    0,
    nrow = p,
    ncol = K
  )
  
  rownames(refit_beta) <- rownames(beta)
  colnames(refit_beta) <- task_names
  
  
  ## Loss contribution
  
  com.loglike <- numeric(K)
  
  
  ## Store score contribution and Hessian
  
  score_contribution <- vector("list", K)
  hessian_list <- vector("list", K)
  
  
  ## Performance measures
  
  MSE <- rep(NA_real_, K)
  MAE <- rep(NA_real_, K)
  
  AUC <- rep(NA_real_, K)
  DEV <- rep(NA_real_, K)
  
  AFT_loss <- rep(NA_real_, K)
  Event_rate <- rep(NA_real_, K)
  
  
  ## ============================================================
  ## 6. Small AUC helper
  ## ============================================================
  
  .auc_rank <- function(y, prob) {
    
    keep <- is.finite(y) & is.finite(prob)
    
    y <- y[keep]
    prob <- prob[keep]
    
    n1 <- sum(y == 1)
    n0 <- sum(y == 0)
    
    if (n1 == 0 || n0 == 0) {
      return(NA_real_)
    }
    
    r <- rank(prob)
    
    auc <- (
      sum(r[y == 1]) -
        n1 * (n1 + 1) / 2
    ) / (n1 * n0)
    
    auc
  }
  
  
  ## ============================================================
  ## 7. Fit each task separately using selected variables
  ## ============================================================
  
  for (k in seq_len(K)) {
    
    task <- tasks_work[[k]]
    
    type_k <- task$type
    
    design <- as.matrix(task$x[, feature, drop = FALSE])
    
    nk <- nrow(design)
    
    
    ## ----------------------------------------------------------
    ## A. Standard regression
    ## ----------------------------------------------------------
    
    if (type_k == "regression") {
      
      yk <- as.numeric(task$y)
      
      fit <- lm.fit(
        x = design,
        y = yk
      )
      
      coef_k <- fit$coefficients
      
      coef_k[is.na(coef_k)] <- 0
      
      fitted_k <- as.vector(
        design %*% coef_k
      )
      
      residual_k <- yk - fitted_k
      
      score_resid_k <- residual_k
      
      
      ## mean squared loss / 2
      
      loss_k <- mean(residual_k^2) / 2
      
      
      MSE[k] <- mean(residual_k^2)
      
      MAE[k] <- mean(abs(residual_k))
      
      
      ## Hessian of squared loss
      
      Hk <- crossprod(design) / nk
      
      
      refit_beta[feature, k] <- coef_k
      
      residuals[1:nk, k] <- residual_k
      
      score_residuals[1:nk, k] <- score_resid_k
      
      com.loglike[k] <- 2 * loss_k
      
      
      ## ----------------------------------------------------------
      ## B. Huber regression
      ## ----------------------------------------------------------
      
    } else if (
      type_k %in%
      c("Huber regression", "adaptive Huber")
    ) {
      
      yk <- as.numeric(task$y)
      
      if (type_k == "adaptive Huber" && !is.null(tau_effective)) {
        tau_k <- tau_effective[k]
      } else {
        tau_k <- task$tau
        
        if (is.null(tau_k)) {
          tau_k <- 1.45
        }
      }
      
      if (length(tau_k) != 1L || !is.finite(tau_k) || tau_k <= 0) {
        stop(paste0(
          "A positive finite Huber threshold is required for task ", k, "."
        ))
      }
      
      
      beta_start <- beta[feature, k]
      
      
      fit <- optim(
        par = beta_start,
        
        fn = function(b) {
          
          .cont_loss(
            design,
            yk,
            b,
            tau = tau_k,
            method = type_k
          )[[2]]
          
        },
        
        gr = function(b) {
          
          .cont_loss(
            design,
            yk,
            b,
            tau = tau_k,
            method = type_k
          )[[1]]
          
        },
        
        method = "BFGS",
        
        control = list(
          maxit = max_iter,
          reltol = tol
        )
      )
      
      if (fit$convergence != 0L) {
        warning(paste0(
          "Huber refit did not fully converge for task ", k,
          ": ", fit$message
        ))
      }
      
      
      coef_k <- fit$par
      
      
      out <- .cont_loss(
        design,
        yk,
        coef_k,
        tau = tau_k,
        method = type_k
      )
      
      
      loss_k <- out[[2]]
      
      tau_final <- out[[3]]
      
      residual_k <- out[[4]]
      
      score_resid_k <- out[[5]]
      
      
      fitted_k <- yk - residual_k
      
      
      MSE[k] <- mean(residual_k^2)
      
      MAE[k] <- mean(abs(residual_k))
      
      
      ## Hessian approximation for Huber loss
      ##
      ## observations inside Huber threshold contribute
      ## quadratic curvature
      
      huber_weight <- as.numeric(
        abs(residual_k) <= tau_final
      )
      
      Hk <- crossprod(
        design,
        design * huber_weight
      ) / nk
      
      
      refit_beta[feature, k] <- coef_k
      
      residuals[1:nk, k] <- residual_k
      
      score_residuals[1:nk, k] <- score_resid_k
      
      com.loglike[k] <- 2 * loss_k
      
      
      ## ----------------------------------------------------------
      ## C. Logistic regression
      ## ----------------------------------------------------------
      
    } else if (
      type_k == "logistic regression"
    ) {
      
      yk <- as.numeric(task$y)
      
      
      fit <- glm.fit(
        x = design,
        y = yk,
        family = binomial(link = "logit")
      )
      
      
      coef_k <- fit$coefficients
      
      coef_k[!is.finite(coef_k)] <- 0
      
      
      eta <- as.vector(
        design %*% coef_k
      )
      
      prob <- plogis(eta)
      
      
      ## keep away from exactly 0/1
      
      prob_safe <- pmin(
        pmax(prob, 1e-12),
        1 - 1e-12
      )
      
      
      residual_k <- yk - prob
      
      score_resid_k <- residual_k
      
      
      loss_k <- -mean(
        yk * log(prob_safe) +
          (1 - yk) * log(1 - prob_safe)
      )
      
      
      DEV[k] <- -2 * sum(
        yk * log(prob_safe) +
          (1 - yk) * log(1 - prob_safe)
      )
      
      
      AUC[k] <- .auc_rank(
        yk,
        prob
      )
      
      
      ## Logistic Hessian
      
      w <- prob * (1 - prob)
      
      Hk <- crossprod(
        design,
        design * w
      ) / nk
      
      
      refit_beta[feature, k] <- coef_k
      
      residuals[1:nk, k] <- residual_k
      
      score_residuals[1:nk, k] <- score_resid_k
      
      com.loglike[k] <- 2 * loss_k
      
      
      ## ----------------------------------------------------------
      ## D. Exponential AFT
      ## ----------------------------------------------------------
      
    } else if (
      type_k == "AFT exponential"
    ) {
      
      ## The same format is used by .Main_Algo():
      ## task$y is observed time and task$delta is event status.
      
      time_k <- as.numeric(task$y)
      
      delta_k <- as.numeric(task$delta)
      
      
      beta_start <- beta[feature, k]
      
      
      fit <- optim(
        par = beta_start,
        
        fn = function(b) {
          
          .aft_exp_loss(
            design,
            time_k,
            delta_k,
            b
          )[[2]]
          
        },
        
        gr = function(b) {
          
          .aft_exp_loss(
            design,
            time_k,
            delta_k,
            b
          )[[1]]
          
        },
        
        method = "BFGS",
        
        control = list(
          maxit = max_iter,
          reltol = tol
        )
      )
      
      if (fit$convergence != 0L) {
        warning(paste0(
          "AFT refit did not fully converge for task ", k,
          ": ", fit$message
        ))
      }
      
      
      coef_k <- fit$par
      
      
      out <- .aft_exp_loss(
        design,
        time_k,
        delta_k,
        coef_k
      )
      
      
      loss_k <- out[[2]]
      
      score_resid_k <- out[[3]]
      
      hazard_k <- out[[4]]
      
      eta <- out[[5]]
      
      
      ## For AFT the returned residual is a SCORE residual:
      ##
      ## delta - time * exp(-eta)
      ##
      ## It should NOT be interpreted as y - fitted value.
      
      residual_k <- score_resid_k
      
      
      AFT_loss[k] <- loss_k
      
      Event_rate[k] <- mean(delta_k)
      
      
      ## Hessian:
      ##
      ## time * exp(-eta)
      
      aft_weight <- time_k * hazard_k
      
      Hk <- crossprod(
        design,
        design * aft_weight
      ) / nk
      
      
      refit_beta[feature, k] <- coef_k
      
      residuals[1:nk, k] <- residual_k
      
      score_residuals[1:nk, k] <- score_resid_k
      
      com.loglike[k] <- 2 * loss_k
      
      
      ## ----------------------------------------------------------
      ## Unknown task
      ## ----------------------------------------------------------
      
    } else {
      
      stop(
        paste0(
          "Unknown task type for task ",
          k,
          ": ",
          type_k
        )
      )
      
    }
    
    
    ## ========================================================
    ## individual score contribution
    ##
    ## X_i * score residual_i
    ## ========================================================
    
    score_contribution[[k]] <-
      design * score_resid_k
    
    
    hessian_list[[k]] <- Hk
    
  }
  
  
  ## ============================================================
  ## 8. Sandwich-type effective degree of freedom
  ## ============================================================
  
  H <- as.matrix(
    Matrix::bdiag(hessian_list)
  )
  
  
  ## ----------------------------------------------------------
  ## J matrix
  ##
  ## paired_tasks = TRUE:
  ## allow covariance between score contributions from different
  ## tasks, provided observations correspond across tasks.
  ##
  ## paired_tasks = FALSE:
  ## block-diagonal covariance.
  ## ----------------------------------------------------------
  
  if (
    paired_tasks &&
    length(unique(n)) == 1
  ) {
    
    score_all <- do.call(
      cbind,
      score_contribution
    )
    
    J <- cov(score_all)
    
    
  } else {
    
    J_list <- lapply(
      score_contribution,
      function(z) {
        
        if (nrow(z) <= 1) {
          
          return(
            matrix(
              0,
              nrow = ncol(z),
              ncol = ncol(z)
            )
          )
          
        }
        
        cov(z)
      }
    )
    
    
    J <- as.matrix(
      Matrix::bdiag(J_list)
    )
    
  }
  
  
  ## ----------------------------------------------------------
  ## Stabilize Hessian before inversion
  ## ----------------------------------------------------------
  
  ridge <- 1e-8
  
  H_stable <- H +
    diag(ridge, nrow(H))
  
  
  degree_of_freedom <- tryCatch(
    
    {
      
      sum(
        diag(
          J %*%
            solve(H_stable)
        )
      )
      
    },
    
    error = function(e) {
      
      warning(
        "Degree-of-freedom calculation failed."
      )
      
      NA_real_
      
    }
    
  )
  
  
  ## ============================================================
  ## 9. Composite information criterion
  ## ============================================================
  
  compo_likelihood <- sum(
    com.loglike * import_w
  )
  
  
  if (is.finite(degree_of_freedom)) {
    
    df_penalty <-
      alpha *
      degree_of_freedom *
      log(max(p1, 2))
    
    info_criterion <-
      compo_likelihood +
      df_penalty
    
  } else {
    
    info_criterion <- NA_real_
    
  }
  
  
  Bayesian_Information <- c(
    "Compo_likelihood" = compo_likelihood,
    "Degree of freedom" = degree_of_freedom,
    "Info criterion" = info_criterion
  )
  
  
  ## ============================================================
  ## 10. Performance tables
  ## ============================================================
  
  ## ----------------------------------------------------------
  ## Regression tasks
  ## ----------------------------------------------------------
  
  reg_id <- which(
    type %in%
      c(
        "regression",
        "Huber regression",
        "adaptive Huber"
      )
  )
  
  
  Reg_Error <- NULL
  
  if (length(reg_id) > 0) {
    
    Reg_Error <- rbind(
      MSE = MSE[reg_id],
      MAE = MAE[reg_id]
    )
    
    colnames(Reg_Error) <-
      task_names[reg_id]
    
  }
  
  
  ## ----------------------------------------------------------
  ## Logistic tasks
  ## ----------------------------------------------------------
  
  logit_id <- which(
    type == "logistic regression"
  )
  
  
  Class_Perform <- NULL
  
  if (length(logit_id) > 0) {
    
    Class_Perform <- rbind(
      AUC = AUC[logit_id],
      DEV = DEV[logit_id]
    )
    
    colnames(Class_Perform) <-
      task_names[logit_id]
    
  }
  
  
  ## ----------------------------------------------------------
  ## Survival tasks
  ## ----------------------------------------------------------
  
  aft_id <- which(
    type == "AFT exponential"
  )
  
  
  Surv_Perform <- NULL
  
  if (length(aft_id) > 0) {
    
    Surv_Perform <- rbind(
      AFT_Loss = AFT_loss[aft_id],
      Event_Rate = Event_rate[aft_id]
    )
    
    colnames(Surv_Perform) <-
      task_names[aft_id]
    
  }
  
  
  ## ============================================================
  ## 11. Overall task summary
  ## ============================================================
  
  Task_Performance <- data.frame(
    
    Task = seq_len(K),
    
    Task_Name = task_names,
    
    Type = type,
    
    N = n,
    
    MSE = MSE,
    
    MAE = MAE,
    
    AUC = AUC,
    
    Deviance = DEV,
    
    AFT_Loss = AFT_loss,
    
    Event_Rate = Event_rate,
    
    Composite_Loss =
      com.loglike,
    
    stringsAsFactors = FALSE
    
  )
  
  
  ## ============================================================
  ## 12. Return results
  ## ============================================================
  
  Results <- list(
    
    Bayesian_Information =
      Bayesian_Information,
    
    Selected_Features =
      feature,
    
    Refit_beta =
      refit_beta,
    
    Task_Performance =
      Task_Performance,
    
    Reg_Error =
      Reg_Error,
    
    Class_Perform =
      Class_Perform,
    
    Surv_Perform =
      Surv_Perform,
    
    Residuals =
      residuals,
    
    Score_Residuals =
      score_residuals
    
  )
  
  
  return(Results)
  
}


.prepare_mtl_tasks <- function(
    tasks,
    task_weights = NULL,
    complete = TRUE,
    missing_threshold = 0.10
) {
  
  if (!is.list(tasks) || length(tasks) == 0L) {
    stop("tasks must be a non-empty list.")
  }
  
  K <- length(tasks)
  
  if (!is.logical(complete) || length(complete) != 1L || is.na(complete)) {
    stop("complete must be TRUE or FALSE.")
  }
  
  if (length(missing_threshold) != 1L ||
      !is.finite(missing_threshold) ||
      missing_threshold < 0 || missing_threshold > 1) {
    stop("missing_threshold must be between 0 and 1.")
  }
  
  if (!is.null(task_weights)) {
    if (length(task_weights) == 1L) {
      task_weights <- rep(task_weights, K)
    }
    
    if (length(task_weights) != K || anyNA(task_weights) ||
        any(!is.finite(task_weights)) || any(task_weights <= 0)) {
      stop("task_weights must contain K positive finite values.")
    }
  }
  
  tasks_work <- tasks
  p <- NULL
  missing_proportion <- NULL
  
  for (k in seq_len(K)) {
    task <- tasks_work[[k]]
    
    if (is.null(task$x) || is.null(task$y) || is.null(task$type)) {
      stop(paste0("Task ", k, " must contain x, y, and type."))
    }
    
    task$x <- as.matrix(task$x)
    storage.mode(task$x) <- "double"
    
    if (is.null(p)) {
      p <- ncol(task$x)
      
      if (is.null(p) || p < 1L) {
        stop("Every task must contain at least one predictor.")
      }
      
      missing_proportion <- matrix(
        0,
        nrow = p,
        ncol = K,
        dimnames = list(colnames(task$x), names(tasks))
      )
    }
    
    if (ncol(task$x) != p) {
      stop("All tasks must contain the same number of predictors.")
    }
    
    if (any(is.infinite(task$x))) {
      stop(paste0("Infinite predictor values were found in task ", k, "."))
    }
    
    if (isTRUE(complete)) {
      if (anyNA(task$x)) {
        stop(paste0(
          "Missing predictor values were found in task ", k,
          ". Use complete = FALSE to apply the package imputation rule."
        ))
      }
    } else {
      for (j in seq_len(p)) {
        missing_j <- is.na(task$x[, j])
        proportion_j <- mean(missing_j)
        missing_proportion[j, k] <- proportion_j
        
        if (!any(missing_j)) {
          next
        }
        
        if (proportion_j <= missing_threshold && any(!missing_j)) {
          task$x[missing_j, j] <- mean(task$x[!missing_j, j])
        } else {
          task$x[, j] <- 0
        }
      }
    }
    
    if (!is.null(task_weights)) {
      task$weight <- task_weights[k]
    }
    
    tasks_work[[k]] <- task
  }
  
  list(
    tasks = tasks_work,
    missing_proportion = missing_proportion
  )
}