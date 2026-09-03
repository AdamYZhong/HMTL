#' Heterogeneous Multi-Task Feature Learning
#'
#' Fits continuous, binary, survival, or mixed-outcome multitask models with
#' one accelerated proximal-gradient algorithm. Each task selects its own loss
#' through its `type` field. A row-wise group-lasso penalty provides common
#' feature selection, and an optional similarity penalty encourages selected
#' groups of tasks to have similar coefficients.
#'
#' @param tasks A non-empty list of task specifications. Every task must contain
#'   `x`, `y`, and `type`. Allowed types are `"regression"`,
#'   `"Huber regression"`, `"adaptive Huber"`,
#'   `"logistic regression"`, and `"AFT exponential"`. An AFT task must
#'   additionally contain `delta`, a 0/1 event indicator. A Huber task may
#'   contain `tau`. A task may contain `weight`.
#' @param lambda1 Nonnegative row-wise group-lasso penalty parameter.
#' @param lambda2 Nonnegative within-task-group similarity penalty parameter.
#'   The default is zero, which excludes this penalty.
#' @param task_groups Optional list of non-overlapping task-index vectors.
#'   Required when `lambda2 > 0`.
#' @param beta Initial coefficient value or a p by K coefficient matrix.
#' @param task_weights Optional scalar or length-K vector of positive task
#'   weights. When supplied, these values override `task$weight`.
#' @param standardize Logical; whether each task's predictors are divided by
#'   their task-specific standard deviations during optimization.
#' @param complete Logical. If `TRUE`, missing predictors cause an error. If
#'   `FALSE`, columns with missingness no greater than `missing_threshold` are
#'   mean-imputed; columns exceeding the threshold are set to zero.
#' @param missing_threshold Missingness threshold used when `complete = FALSE`.
#' @param adaptive_divisor Positive scaling divisor used to determine the
#'   adaptive-Huber threshold.
#' @param tol Positive convergence tolerance.
#' @param selection_tol Nonnegative threshold used to identify nonzero rows.
#' @param max_iter Maximum number of proximal-gradient iterations.
#' @param max_backtrack Maximum number of backtracking steps per iteration.
#' @param accelerate Logical; whether to use FISTA acceleration.
#' @param diagnostics Logical; whether to refit the selected support and return
#'   information-criterion and task-performance results.
#' @param alpha Nonnegative multiplier for the information-criterion complexity
#'   penalty.
#' @param paired_tasks Logical. If `TRUE` and all task sample sizes agree,
#'   cross-task score covariance is used in the effective degrees of freedom.
#'   Rows must represent the same subjects in the same order.
#' @param verbose Logical; whether to print iteration updates.
#'
#' @return An object of classes `mtl_fit` and `mixed_mtl_fit`. It contains the
#'   sparse estimates, original-scale estimates, selected variables, losses,
#'   convergence history, and—when requested—diagnostic results.
#'
#' @details The optimized objective is
#' \deqn{\sum_{k=1}^K w_k L_k(\beta_k)
#' + \lambda_1 \sum_{j=1}^p \|\beta_{j\cdot}\|_2
#' + \lambda_2 \sum_g \sum_{k\in G_g}\|\beta_k-B_g\|_2^2,}
#' where `L_k` is determined by the type of task k and `B_g` is the mean
#' coefficient vector among the tasks in group g.
#'
#' @examples
#' \dontrun{
#' tasks <- list(
#'   continuous = list(x = x1, y = y1, type = "adaptive Huber", tau = 1.45),
#'   binary = list(x = x2, y = y2, type = "logistic regression"),
#'   survival = list(
#'     x = x3, y = time3, delta = status3, type = "AFT exponential"
#'   )
#' )
#'
#' fit <- MTL(tasks, lambda1 = 0.1, diagnostics = TRUE)
#' fit$beta_original_scale
#' fit$Bayesian_Information
#' }
#'
#' @export
MTL <- function(
    tasks,
    lambda1,
    lambda2 = 0,
    task_groups = NULL,
    beta = 0.1,
    task_weights = NULL,
    standardize = TRUE,
    complete = TRUE,
    missing_threshold = 0.10,
    adaptive_divisor = 10,
    tol = 1e-5,
    selection_tol = 1e-10,
    max_iter = 500,
    max_backtrack = 30,
    accelerate = TRUE,
    diagnostics = FALSE,
    alpha = 1,
    paired_tasks = FALSE,
    verbose = FALSE
) {
  
  if (!is.logical(diagnostics) ||
      length(diagnostics) != 1L || is.na(diagnostics)) {
    stop("diagnostics must be TRUE or FALSE.")
  }
  
  prepared <- .prepare_mtl_tasks(
    tasks = tasks,
    task_weights = task_weights,
    complete = complete,
    missing_threshold = missing_threshold
  )
  
  tasks_work <- prepared$tasks
  
  fit <- .Main_Algo(
    tasks = tasks_work,
    lambda1 = lambda1,
    lambda2 = lambda2,
    task_groups = task_groups,
    beta = beta,
    standardize = standardize,
    adaptive_divisor = adaptive_divisor,
    tol = tol,
    selection_tol = selection_tol,
    max_iter = max_iter,
    max_backtrack = max_backtrack,
    accelerate = accelerate,
    verbose = verbose
  )
  
  fit$missing_proportion <- prepared$missing_proportion
  fit$diagnostics_requested <- diagnostics
  fit$package_call <- match.call()
  
  # Compatibility aliases for code written against earlier package output.
  fit$Selected_List <- fit$selected_variables
  fit$Task_type <- fit$task_information$type
  fit$Task_weights <- fit$task_weights
  
  if (isTRUE(diagnostics)) {
    diagnostic_results <- .info_fitting(
      tasks = tasks_work,
      beta = fit,
      alpha = alpha,
      tol = tol,
      max_iter = max_iter,
      selection_tol = selection_tol,
      paired_tasks = paired_tasks
    )
    
    fit$diagnostics <- diagnostic_results
    
    # Keep the principal diagnostic elements at the top level for convenient
    # access and compatibility with earlier package output.
    fit[names(diagnostic_results)] <- diagnostic_results
  } else {
    fit$diagnostics <- NULL
  }
  
  class(fit) <- unique(c("mtl_fit", class(fit)))
  fit
}


#' Information-Criterion Selection for Unified Multi-Task Learning
#'
#' Fits the unified multitask model over candidate values of `lambda1` and
#' optionally `lambda2`, then selects the candidate with the smallest
#' information criterion returned by `.info_fitting()`.
#'
#' @inheritParams MTL
#' @param lambda1 A non-empty vector of candidate group-lasso parameters.
#' @param lambda2 A non-empty vector of candidate similarity parameters.
#' @param keep_fits Logical; whether to retain every fitted model. The selected
#'   best fit is always returned.
#'
#' @return An object of class `mtl_selection` containing `Selection_Results`,
#'   the best tuning parameters, and the corresponding fitted model.
#'
#' @examples
#' \dontrun{
#' selection <- Selection_MTL(
#'   tasks = tasks,
#'   lambda1 = c(0.05, 0.1, 0.2),
#'   lambda2 = 0
#' )
#' selection$Best_lambda1
#' plot(selection)
#' }
#'
#' @export
Selection_MTL <- function(
    tasks,
    lambda1,
    lambda2 = 0,
    task_groups = NULL,
    beta = 0.1,
    task_weights = NULL,
    standardize = TRUE,
    complete = TRUE,
    missing_threshold = 0.10,
    adaptive_divisor = 10,
    tol = 1e-5,
    selection_tol = 1e-10,
    max_iter = 500,
    max_backtrack = 30,
    accelerate = TRUE,
    alpha = 1,
    paired_tasks = FALSE,
    keep_fits = FALSE,
    verbose = FALSE
) {
  
  if (length(lambda1) < 1L || anyNA(lambda1) ||
      any(!is.finite(lambda1)) || any(lambda1 < 0)) {
    stop("lambda1 must contain nonnegative finite candidate values.")
  }
  
  if (length(lambda2) < 1L || anyNA(lambda2) ||
      any(!is.finite(lambda2)) || any(lambda2 < 0)) {
    stop("lambda2 must contain nonnegative finite candidate values.")
  }
  
  if (any(lambda2 > 0) && is.null(task_groups)) {
    stop("task_groups must be supplied when any lambda2 candidate is positive.")
  }
  
  if (!is.logical(keep_fits) ||
      length(keep_fits) != 1L || is.na(keep_fits)) {
    stop("keep_fits must be TRUE or FALSE.")
  }
  
  prepared <- .prepare_mtl_tasks(
    tasks = tasks,
    task_weights = task_weights,
    complete = complete,
    missing_threshold = missing_threshold
  )
  
  tasks_work <- prepared$tasks
  
  candidates <- expand.grid(
    lambda1 = sort(unique(as.numeric(lambda1)), decreasing = TRUE),
    lambda2 = sort(unique(as.numeric(lambda2))),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  
  M <- nrow(candidates)
  
  selection_results <- data.frame(
    Lambda1 = candidates$lambda1,
    Lambda2 = candidates$lambda2,
    Compo_likelihood = NA_real_,
    Degree_of_freedom = NA_real_,
    Info_criterion = NA_real_,
    Selected_features = NA_integer_,
    Converged = FALSE,
    Error = NA_character_,
    stringsAsFactors = FALSE
  )
  
  fitted_models <- if (isTRUE(keep_fits)) vector("list", M) else NULL
  successful_fits <- vector("list", M)
  
  for (i in seq_len(M)) {
    if (isTRUE(verbose)) {
      message(
        "Candidate ", i, "/", M,
        ": lambda1 = ", signif(candidates$lambda1[i], 6),
        ", lambda2 = ", signif(candidates$lambda2[i], 6)
      )
    }
    
    fit_i <- tryCatch(
      MTL(
        tasks = tasks_work,
        lambda1 = candidates$lambda1[i],
        lambda2 = candidates$lambda2[i],
        task_groups = task_groups,
        beta = beta,
        standardize = standardize,
        complete = TRUE,
        adaptive_divisor = adaptive_divisor,
        tol = tol,
        selection_tol = selection_tol,
        max_iter = max_iter,
        max_backtrack = max_backtrack,
        accelerate = accelerate,
        diagnostics = TRUE,
        alpha = alpha,
        paired_tasks = paired_tasks,
        verbose = FALSE
      ),
      error = function(e) e
    )
    
    if (inherits(fit_i, "error")) {
      selection_results$Error[i] <- conditionMessage(fit_i)
      next
    }
    
    bic_i <- fit_i$Bayesian_Information
    
    selection_results$Compo_likelihood[i] <-
      unname(bic_i["Compo_likelihood"])
    selection_results$Degree_of_freedom[i] <-
      unname(bic_i["Degree of freedom"])
    selection_results$Info_criterion[i] <-
      unname(bic_i["Info criterion"])
    selection_results$Selected_features[i] <-
      length(fit_i$selected_variables)
    selection_results$Converged[i] <- isTRUE(fit_i$converged)
    
    successful_fits[[i]] <- fit_i
    
    if (isTRUE(keep_fits)) {
      fitted_models[[i]] <- fit_i
    }
  }
  
  valid <- which(is.finite(selection_results$Info_criterion))
  
  if (length(valid) == 0L) {
    stop(
      "No candidate produced a finite information criterion. ",
      "Inspect the Error column and the task data."
    )
  }
  
  best_index <- valid[
    which.min(selection_results$Info_criterion[valid])
  ]
  
  result <- list(
    Selection_Results = selection_results,
    Best_index = best_index,
    Best_lambda1 = selection_results$Lambda1[best_index],
    Best_lambda2 = selection_results$Lambda2[best_index],
    Best_fit = successful_fits[[best_index]],
    Fits = fitted_models,
    call = match.call()
  )
  
  class(result) <- "mtl_selection"
  result
}


#' Plot Unified MTL Model-Selection Results
#'
#' @param x An object returned by `Selection_MTL()`.
#' @param ... Additional graphical arguments passed to `plot()`.
#'
#' @return The input object, invisibly.
#'
#' @method plot mtl_selection
#' @export
plot.mtl_selection <- function(x, ...) {
  
  results <- x$Selection_Results
  results <- results[is.finite(results$Info_criterion), , drop = FALSE]
  
  if (nrow(results) == 0L) {
    stop("No finite information-criterion values are available to plot.")
  }
  
  lambda2_values <- unique(results$Lambda2)
  
  if (length(lambda2_values) == 1L) {
    graphics::plot(
      results$Lambda1,
      results$Info_criterion,
      type = "b",
      xlab = expression(lambda[1]),
      ylab = "Information criterion",
      main = "Unified MTL model selection",
      ...
    )
  } else {
    y_range <- range(results$Info_criterion, finite = TRUE)
    colors <- seq_along(lambda2_values)
    
    first <- results$Lambda2 == lambda2_values[1]
    
    graphics::plot(
      results$Lambda1[first],
      results$Info_criterion[first],
      type = "b",
      col = colors[1],
      ylim = y_range,
      xlab = expression(lambda[1]),
      ylab = "Information criterion",
      main = "Unified MTL model selection",
      ...
    )
    
    if (length(lambda2_values) > 1L) {
      for (j in 2:length(lambda2_values)) {
        keep <- results$Lambda2 == lambda2_values[j]
        graphics::lines(
          results$Lambda1[keep],
          results$Info_criterion[keep],
          type = "b",
          col = colors[j]
        )
      }
    }
    
    graphics::legend(
      "topright",
      legend = paste0("lambda2 = ", lambda2_values),
      col = colors,
      lty = 1,
      pch = 1,
      bty = "n"
    )
  }
  
  invisible(x)
}


#' Backward-Compatible Selection Plot
#'
#' This alias allows existing calls to `plot_HMTL()` to display results from
#' the new unified `Selection_MTL()` function.
#'
#' @param x An object returned by `Selection_MTL()`.
#' @param ... Additional graphical arguments.
#'
#' @return The input object, invisibly.
#'
#' @export
plot_HMTL <- function(x, ...) {
  if (!inherits(x, "mtl_selection")) {
    stop("x must be an object returned by Selection_MTL().")
  }
  
  graphics::plot(x, ...)
}


