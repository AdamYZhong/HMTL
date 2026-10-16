# HMTL

HMTL provides heterogeneous multi-task feature learning for mixed outcome types.

Supported task types:

- `"regression"`
- `"Huber regression"`
- `"adaptive Huber"`
- `"logistic regression"`
- `"AFT exponential"`

The optimization combines task-specific losses with a row-wise group-lasso penalty for common feature selection and an optional within-task-group squared similarity penalty.



## Basic example

```r
library(HMTL)

tasks <- list(
  continuous = list(x = x1, y = y1, type = "adaptive Huber", tau = 1.45),
  binary = list(x = x2, y = y2, type = "logistic regression"),
  survival = list(x = x3, y = time3, delta = status3, type = "AFT exponential")
)

fit <- MTL(tasks, lambda1 = 0.1, diagnostics = TRUE)
fit$beta_original_scale
fit$selected_names

sel <- Selection_MTL(tasks, lambda1 = c(0.05, 0.1, 0.2))
plot(sel)
```
