#' @title TETSsetup
#' @description Sets up TOBIT TETS general univariate models
#'
#' @details See help of \code{TETSforecast}.
#'
#' @param y a time series to forecast (it may be either a numerical vector or
#' a time series object). This is the only input required. If a vector, the additional
#' input \code{s} should be supplied compulsorily (see below).
#' @param u a matrix of input time series. If 
#' the output wanted to be forecast, matrix \code{u} should contain future values for inputs.
#' @param model the model to estimate. It is a single string indicating the type of 
#' model for each component with one or two letters:
#' \itemize{
#' \item Error: ? / A
#' 
#' \item Trend: ? / N / A / Ad
#' 
#' \item Seasonal: ? / N / A
#' 
#' }
#' @param s seasonal period of time series (1 for annual, 4 for quarterly, ...)
#' @param h forecast horizon. If the model includes inputs h is not used, the lenght of u is used instead.
#' @param criterion information criterion for identification ("aic", "bic" or "aicc").
#' @param forIntervals estimate forecasting intervals (TRUE / FALSE)
#' @param bootstrap use bootstrap simulation for predictive distributions
#' @param nSimul number of simulation runs for bootstrap simulation of predictive distributions
#' @param verbose intermediate estimation output (TRUE / FALSE)
#' @param alphaL constraints limits for alpha parameter
#' @param betaL constraints limits for beta parameter
#' @param gammaL constraints limits for gamma parameter
#' @param phiL constraints limits for phi parameter
#' @param p0 initial values for parameter search (alpha, beta, phi, gamma, sigma2) with consraints:
#' @param Ymin scalar or vector of time varying censoring values from below
#' @param Ymax scalar or vector of time varying censoring values from above
#' 
#' \itemize{ 
#' \item 0 < alpha < 1
#' 
#' \item 0 < beta < alpha
#' 
#' \item 0 < phi < 1
#' 
#' \item 0 < gamma < 1 - alpha
#' 
#' \item sigma2 > 0
#' }
#' 
#' @return An object of class \code{TETS}. It is a list with fields including all the inputs and
#'         the fields listed below as outputs. All the functions in this package fill in
#'         part of the fields of any \code{TETS} object as specified in what follows (function 
#'         \code{TETS} fills in all of them at once):
#' 
#' After running \code{TETSforecast}:
#' \item{p}{Estimated parameters}
#' \item{criteria}{Values for estimation criteria (LogLik, AIC, BIC, AICc)}
#' \item{yFor}{Forecasted values of output}
#' \item{yForV}{Variance of forecasted values of output}
#' \item{ySimul}{Bootstrap simulations for forecasting distribution evaluation}
#' 
#' After running \code{TETSvalidate}:
#' \item{table}{Estimation and validation table}
#' \item{comp}{Estimated components in matrix form}
#' 
#' After running \code{TETScomponents}:
#' \item{comp}{Estimated components in matrix form}
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{TETS}}, \code{\link{TETSforecast}}, \code{\link{TETSvalidate}},
#'          \code{\link{TETScomponents}}
#'          
#' @examples
#' \donttest{
#' y <- log(AirPassengers)
#' m1 <- TETSsetup(y)
#' m1 <- TETSsetup(y,  model = "???")
#' m1 <- TETSsetup(y,  model = "?AA")
#' }
#' @rdname TETSsetup
#' @export
TETSsetup = function(y, u = NULL, model = "???", s = frequency(y), h = 2 * s, criterion = "aicc", 
                    forIntervals = FALSE, bootstrap = FALSE, nSimul = 5000, verbose = FALSE,
                    alphaL = c(0, 1), betaL = alphaL, gammaL = alphaL, 
                    phiL = c(0.8, 0.98), p0 = -99999, Ymin = -Inf, Ymax = Inf){
    out = ETSsetup(y, u, model, s, h, criterion, 1, FALSE, FALSE, forIntervals,
                   bootstrap, nSimul, verbose, alphaL, betaL, gammaL, phiL, p0)
    out$Ymin = Ymin
    out$Ymax = Ymax
    if (min(Ymax - y, na.rm = TRUE) != 0 && min(y - Ymin, na.rm = TRUE) != 0){
        return(structure(out, class = "ETS"))
    } else {
        return(structure(out, class = "TETS"))
    }
    return(out)
}
#' @title TETSforecast
#' @description Estimates and forecasts TOBIT TETS general univariate models
#'
#' @details \code{TETSforecast} is a function for modelling and forecasting univariate
#' time series with TOBIT ExponenTial Smoothing (TETS) time series models. 
#' It sets up the model with a number of control variables that
#' govern the way the rest of functions in the package will work. It also estimates 
#' the model parameters by Maximum Likelihood and forecasts the data.
#'
#' @inheritParams TETS
#' 
#' @return An object of class \code{TETS}. It is a list with fields including all the inputs and
#'         the fields listed below as outputs. All the functions in this package fill in
#'         part of the fields of any \code{TETS} object as specified in what follows (function 
#'         \code{TETS} fills in all of them at once):
#' 
#' After running \code{TETSforecast}:
#' \item{p}{Estimated parameters}
#' \item{criteria}{Values for estimation criteria (LogLik, AIC, BIC, AICc)}
#' \item{yFor}{Forecasted values of output}
#' \item{yForV}{Variance of forecasted values of output}
#' \item{ySimul}{Bootstrap simulations for forecasting distribution evaluation}
#' 
#' After running \code{TETSvalidate}:
#' \item{table}{Estimation and validation table}
#' \item{comp}{Estimated components in matrix form}
#' 
#' After running \code{TETScomponents}:
#' \item{comp}{Estimated components in matrix form}
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{TETS}}, \code{\link{TETSvalidate}},
#'          \code{\link{TETScomponents}}
#'          
#' @examples
#' \donttest{
#' y <- log(AirPassengers)
#' m1 <- TETSforecast(y)
#' m1 <- TETSforecast(y, model = "A?A")
#' }
#' @rdname TETSforecast
#' @export
TETSforecast = function(y, u = NULL, model = "???", s = frequency(y), h = max(2 * s, 6), criterion = "aicc", 
                    forIntervals = FALSE, bootstrap = FALSE, nSimul = 5000, verbose = FALSE,
                    alphaL = c(0, 1), betaL = alphaL, gammaL = alphaL, 
                    phiL = c(0.8, 0.98), p0 = -99999, Ymin = -Inf, Ymax = Inf){
    m1 = TETSsetup(y, u, model, s, h, criterion, forIntervals,
                  bootstrap, nSimul, verbose, alphaL, betaL, gammaL, phiL, p0, Ymin, Ymax)
    if (inherits(m1, "ETS"))
        m1 = ETSestim(m1)
    else
        m1 = TETSestim(m1)
    if (verbose)
            cat(m1$table)
    return(m1)
}
#' @title TETS
#' @description Runs all relevant functions for TETS modelling
#'
#' @details See help of \code{TETSforecast}.
#'
#' @param y a time series to forecast (it may be either a numerical vector or
#' a time series object). This is the only input required. If a vector, the additional
#' input \code{s} should be supplied compulsorily (see below).
#' @param u a matrix of input time series. If 
#' the output wanted to be forecast, matrix \code{u} should contain future values for inputs.
#' @param model the model to estimate. It is a single string indicating the type of 
#' model for each component with one or two letters:
#' \itemize{
#' \item Error: ? / A
#' 
#' \item Trend: ? / N / A / Ad
#' 
#' \item Seasonal: ? / N / A
#' 
#' }
#' @param s seasonal period of time series (1 for annual, 4 for quarterly, ...)
#' @param h forecast horizon. If the model includes inputs h is not used, the lenght of u is used instead.
#' @param criterion information criterion for identification ("aic", "bic" or "aicc").
#' @param forIntervals estimate forecasting intervals (TRUE / FALSE)
#' @param bootstrap use bootstrap simulation for predictive distributions
#' @param nSimul number of simulation runs for bootstrap simulation of predictive distributions
#' @param verbose intermediate estimation output (TRUE / FALSE)
#' @param alphaL constraints limits for alpha parameter
#' @param betaL constraints limits for beta parameter
#' @param gammaL constraints limits for gamma parameter
#' @param phiL constraints limits for phi parameter
#' @param p0 initial values for parameter search (alpha, beta, phi, gamma, sigma2) with consraints:
#' @param Ymin scalar or vector of time varying censoring values from below
#' @param Ymax scalar or vector of time varying censoring values from above
#' 
#' \itemize{ 
#' \item 0 < alpha < 1
#' 
#' \item 0 < beta < alpha
#' 
#' \item 0 < phi < 1
#' 
#' \item 0 < gamma < 1 - alpha
#' 
#' \item sigma2 > 0
#' }
#' 
#' @author Diego J. Pedregal
#' 
#' @return An object of class \code{TETS}. It is a list with fields including all the inputs and
#'         the fields listed below as outputs. All the functions in this package fill in
#'         part of the fields of any \code{TETS} object as specified in what follows (function 
#'         \code{TETS} fills in all of them at once):
#' 
#' After running \code{TETSforecast}:
#' \item{p}{Estimated parameters}
#' \item{criteria}{Values for estimation criteria (LogLik, AIC, BIC, AICc)}
#' \item{yFor}{Forecasted values of output}
#' \item{yForV}{Variance of forecasted values of output}
#' \item{ySimul}{Bootstrap simulations for forecasting distribution evaluation}
#' 
#' After running \code{TETSvalidate}:
#' \item{table}{Estimation and validation table}
#' \item{comp}{Estimated components in matrix form}
#' 
#' After running \code{TETScomponents}:
#' \item{comp}{Estimated components in matrix form}
#' 
#' @seealso \code{\link{TETSforecast}}, \code{\link{TETSvalidate}},
#'          \code{\link{TETScomponents}}
#'          
#' @examples
#' \donttest{
#' y <- log(AirPassengers)
#' m1 <- TETS(y)
#' m1 <- TETS(y, model = "MAM")
#' }
#' @rdname TETS
#' @export
TETS = function(y, u = NULL, model = "???", s = frequency(y), h = 2 * s, criterion = "aicc", 
               forIntervals = FALSE, bootstrap = FALSE, nSimul = 5000, verbose = FALSE,
               alphaL = c(0, 1), betaL = alphaL, gammaL = alphaL, 
               phiL = c(0.8, 0.98), p0 = -99999, Ymin = -Inf, Ymax = Inf){
    m1 = TETSsetup(y, u, model, s, h, criterion, forIntervals,
                  bootstrap, nSimul, verbose, alphaL, betaL, gammaL, phiL, p0, Ymin, Ymax)
    if (inherits(m1, "ETS")){
        m1 = ETSvalidate(m1)
    } else {
        m1 = TETSvalidate(m1)
    }
    m1$v = m1$comp[, 1]
    if (verbose)
            cat(m1$table)
    return(m1)
}
