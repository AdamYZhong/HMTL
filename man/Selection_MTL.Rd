\name{Selection_MTL}
\alias{Selection_MTL}
\title{Information-Criterion Selection for HMTL}
\usage{
Selection_MTL(
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
  alpha = 1,
  paired_tasks = FALSE,
  keep_fits = FALSE,
  verbose = FALSE
)
}
\arguments{
\item{tasks, task_groups, beta, task_weights, standardize, complete, missing_threshold, adaptive_divisor, tol, selection_tol, max_iter, max_backtrack, accelerate, alpha, paired_tasks, verbose}{See \code{\link{MTL}}.}
\item{lambda1}{A non-empty vector of candidate group-lasso parameters.}
\item{lambda2}{A non-empty vector of candidate similarity parameters.}
\item{keep_fits}{Whether to retain every fitted candidate model.}
}
\value{An object of class \code{mtl_selection} containing the candidate table, selected tuning parameters, and best fitted model.}
\description{Fits HMTL over candidate values of \code{lambda1} and \code{lambda2} and selects the candidate with the smallest returned information criterion.}
\examples{
\dontrun{
selection <- Selection_MTL(tasks, lambda1 = c(0.05, 0.1, 0.2), lambda2 = 0)
selection$Best_lambda1
plot(selection)
}
}
