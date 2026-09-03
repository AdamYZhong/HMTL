\name{MTL}
\alias{MTL}
\title{Heterogeneous Multi-Task Feature Learning}
\usage{
MTL(
  tasks,
  lambda1,
  lambda2 = 0,
  task_groups = NULL,
  beta = 0.1,
  task_weights = NULL,
  standardize = TRUE,
  complete = TRUE,
  missing_threshold = 0.1,
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
)
}
\arguments{
\item{tasks}{A non-empty list of task specifications. Each task contains \code{x}, \code{y}, and \code{type}. Allowed types are \code{"regression"}, \code{"Huber regression"}, \code{"adaptive Huber"}, \code{"logistic regression"}, and \code{"AFT exponential"}. AFT tasks also require \code{delta}.}
\item{lambda1}{Nonnegative row-wise group-lasso penalty parameter.}
\item{lambda2}{Nonnegative within-task-group similarity penalty parameter.}
\item{task_groups}{Optional list of non-overlapping task-index vectors; required when \code{lambda2 > 0}.}
\item{beta}{Initial coefficient value or a p by K coefficient matrix.}
\item{task_weights}{Optional scalar or length-K vector of positive task weights.}
\item{standardize}{Whether to divide predictors by task-specific standard deviations during optimization.}
\item{complete}{If \code{TRUE}, missing predictors cause an error; otherwise package imputation rules are applied.}
\item{missing_threshold}{Maximum missing proportion for mean imputation when \code{complete = FALSE}.}
\item{adaptive_divisor}{Positive divisor used when determining the adaptive-Huber threshold.}
\item{tol}{Positive convergence tolerance.}
\item{selection_tol}{Threshold used to identify nonzero coefficient rows.}
\item{max_iter}{Maximum number of proximal-gradient iterations.}
\item{max_backtrack}{Maximum number of backtracking steps per iteration.}
\item{accelerate}{Whether to use FISTA acceleration.}
\item{diagnostics}{Whether to refit selected variables and return information-criterion and task-performance results.}
\item{alpha}{Nonnegative multiplier for the information-criterion complexity penalty.}
\item{paired_tasks}{Whether equal-sized tasks represent the same subjects in the same order for cross-task covariance calculations.}
\item{verbose}{Whether to print iteration updates.}
}
\value{An object inheriting from \code{mtl_fit} and \code{mixed_mtl_fit}, containing sparse and original-scale coefficients, selected variables, losses, convergence history, and optional diagnostics.}
\description{Fits continuous, binary, survival, or mixed-outcome multitask models with an accelerated proximal-gradient algorithm.}
\details{The objective combines weighted task-specific losses with a row-wise group-lasso penalty and, optionally, a squared within-task-group similarity penalty.}
\examples{
\dontrun{
tasks <- list(
  continuous = list(x = x1, y = y1, type = "adaptive Huber", tau = 1.45),
  binary = list(x = x2, y = y2, type = "logistic regression"),
  survival = list(x = x3, y = time3, delta = status3, type = "AFT exponential")
)
fit <- MTL(tasks, lambda1 = 0.1, diagnostics = TRUE)
fit$beta_original_scale
}
}
