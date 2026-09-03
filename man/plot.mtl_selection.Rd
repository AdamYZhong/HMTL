\name{plot.mtl_selection}
\alias{plot.mtl_selection}
\alias{plot_HMTL}
\title{Plot HMTL Model-Selection Results}
\usage{
\method{plot}{mtl_selection}(x, ...)

plot_HMTL(x, ...)
}
\arguments{
\item{x}{An object returned by \code{Selection_MTL()}.}
\item{...}{Additional graphical arguments passed to plotting functions.}
}
\value{The input object, invisibly.}
\description{Plots information-criterion values across candidate tuning parameters. \code{plot_HMTL()} is a backward-compatible alias.}
