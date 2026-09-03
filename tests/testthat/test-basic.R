test_that("MTL fits two simple task types", {
  set.seed(1)
  x1 <- matrix(rnorm(80), 20, 4)
  x2 <- matrix(rnorm(80), 20, 4)
  colnames(x1) <- colnames(x2) <- paste0("x", 1:4)
  y1 <- as.vector(x1[, 1] - 0.5 * x1[, 2] + rnorm(20, sd = 0.2))
  p2 <- plogis(x2[, 1] - x2[, 2])
  y2 <- rbinom(20, 1, p2)

  tasks <- list(
    continuous = list(x = x1, y = y1, type = "regression"),
    binary = list(x = x2, y = y2, type = "logistic regression")
  )

  fit <- MTL(tasks, lambda1 = 0.05, max_iter = 50)
  expect_s3_class(fit, "mtl_fit")
  expect_equal(dim(fit$beta), c(4L, 2L))
  expect_true(is.finite(fit$objective))
})
