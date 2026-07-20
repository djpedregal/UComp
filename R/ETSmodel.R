#' @title ETSsetup
#' @description Sets up ETS general univariate models
#'
#' @details See help of \code{ETSforecast}.
#'
#' @param y a time series to forecast (it may be either a numerical vector or
#' a time series object). This is the only input required. If a vector, the additional
#' input \code{s} should be supplied compulsorily (see below).
#' @param u a matrix of input time series. If 
#' the output wanted to be forecast, matrix \code{u} should contain future values for inputs.
#' @param model the model to estimate. It is a single string indicating the type of 
#' model for each component with one or two letters:
#' \itemize{
#' \item Error: ? / A / M
#' 
#' \item Trend: ? / N / A / Ad / M / Md 
#' 
#' \item Seasonal: ? / N / A / M
#' 
#' }
#' 
#' @param s seasonal period of time series (1 for annual, 4 for quarterly, ...)
#' @param h forecast horizon. If the model includes inputs h is not used, the lenght of u is used instead.
#' @param criterion information criterion for identification ("aic", "bic" or "aicc").
#' @param lambda Box-Cox lambda parameter (NULL: estimate)
#' @param armaIdent check for arma models for error component (TRUE / FALSE).
#' @param identAll run all models to identify the best one (TRUE / FALSE)
#' @param forIntervals estimate forecasting intervals (TRUE / FALSE)
#' @param bootstrap use bootstrap simulation for predictive distributions
#' @param nSimul number of simulation runs for bootstrap simulation of predictive distributions
#' @param verbose intermediate estimation output (TRUE / FALSE)
#' @param alphaL constraints limits for alpha parameter
#' @param betaL constraints limits for beta parameter
#' @param gammaL constraints limits for gamma parameter
#' @param phiL constraints limits for phi parameter
#' @param p0 initial values for parameter search (alpha, beta, phi, gamma) with consraints:
#' \itemize{ 
#' \item 0 < alpha < 1
#' 
#' \item 0 < beta < alpha
#' 
#' \item 0 < phi < 1
#' 
#' \item 0 < gamma < 1 - alpha
#' }
#' 
#' @return An object of class \code{ETS}. It is a list with fields including all the inputs and
#'         the fields listed below as outputs. All the functions in this package fill in
#'         part of the fields of any \code{ETS} object as specified in what follows (function 
#'         \code{ETS} fills in all of them at once):
#' 
#' After running \code{ETSforecast}:
#' \item{p}{Estimated parameters}
#' \item{criteria}{Values for estimation criteria (LogLik, AIC, BIC, AICc)}
#' \item{yFor}{Forecasted values of output}
#' \item{yForV}{Variance of forecasted values of output}
#' \item{ySimul}{Bootstrap simulations for forecasting distribution evaluation}
#' 
#' After running \code{ETSvalidate}:
#' \item{table}{Estimation and validation table}
#' \item{comp}{Estimated components in matrix form}
#' 
#' After running \code{ETScomponents}:
#' \item{comp}{Estimated components in matrix form}
#' 
#' @author Diego J. Pedregal
#' 
#' @return An object of class \code{ETS}. See \code{ETSforecast}.
#' 
#' @seealso \code{\link{ETS}}, \code{\link{ETSforecast}}, \code{\link{ETSvalidate}},
#'          \code{\link{ETScomponents}}
#'          
#' @examples
#' \donttest{
#' y <- log(AirPassengers)
#' m1 <- ETSsetup(y)
#' m1 <- ETSsetup(y,  model = "???")
#' m1 <- ETSsetup(y,  model = "?AA")
#' }
#' @rdname ETSsetup
#' @export
ETSsetup = function(y, u = NULL, model = "???", s = frequency(y), h = 2 * s, criterion = "aicc", 
                    lambda = 1, armaIdent = FALSE, identAll = FALSE, forIntervals = FALSE,
                    bootstrap = FALSE, nSimul = 5000, verbose = FALSE,
                    alphaL = c(1e-8, 1-1e-8), betaL = alphaL, gammaL = alphaL, 
                    phiL = c(0.8, 0.98), p0 = -99999){
    y = as.ts(y)
    if (alphaL[1] >= alphaL[2])
        stop("Wrong alpha limits!!")
    if (betaL[1] >= betaL[2])
        stop("Wrong alpha limits!!")
    if (gammaL[1] >= gammaL[2])
        stop("Wrong alpha limits!!")
    if (phiL[1] >= phiL[2])
        stop("Wrong alpha limits!!")
    if (s > 24)
        stop("Data with period greater than 24 are not allowed!!")
    if (length(y) < s + 2 || length(y) < 8)
        stop("Error: Not enough data to estimate model!!")
    if (is.null(lambda))
        lambda = 9999.9
    out =  list(y = y,
                u = u,
                model = model,
                s = s,
                h = h,
                p0 = p0,
                criterion = criterion,
                lambda = lambda,
                armaIdent = armaIdent,
                identAll = identAll,
                forIntervals = forIntervals,
                bootstrap = bootstrap,
                nSimul = nSimul,
                verbose = verbose,
                alphaL = alphaL,
                betaL = betaL,
                gammaL = gammaL,
                phiL = phiL,
                # outputs
                criteria = NULL,
                yFor = NULL,
                yForV = NULL,
                comp = NULL,
                ySimul = NULL,
                table = "",
                p = NULL,
                truep = NULL,
                v = NULL)
    return(structure(out, class = "ETS"))
}
#' @title ETSforecast
#' @description Estimates and forecasts ETS general univariate models
#'
#' @details \code{ETSforecast} is a function for modelling and forecasting univariate
#' time series with ExponenTial Smoothing (ETS) time series models. 
#' It sets up the model with a number of control variables that
#' govern the way the rest of functions in the package will work. It also estimates 
#' the model parameters by Maximum Likelihood and forecasts the data.
#'
#' @inheritParams ETS
#' 
#' @return An object of class \code{ETS}. It is a list with fields including all the inputs and
#'         the fields listed below as outputs. All the functions in this package fill in
#'         part of the fields of any \code{ETS} object as specified in what follows (function 
#'         \code{ETS} fills in all of them at once):
#' 
#' After running \code{ETSforecast}:
#' \item{p}{Estimated parameters}
#' \item{criteria}{Values for estimation criteria (LogLik, AIC, BIC, AICc)}
#' \item{yFor}{Forecasted values of output}
#' \item{yForV}{Variance of forecasted values of output}
#' \item{ySimul}{Bootstrap simulations for forecasting distribution evaluation}
#' 
#' After running \code{ETSvalidate}:
#' \item{table}{Estimation and validation table}
#' \item{comp}{Estimated components in matrix form}
#' 
#' After running \code{ETScomponents}:
#' \item{comp}{Estimated components in matrix form}
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{ETS}}, \code{\link{ETSvalidate}},
#'          \code{\link{ETScomponents}}
#'          
#' @examples
#' \donttest{
#' y <- log(AirPassengers)
#' m1 <- ETSforecast(y)
#' m1 <- ETSforecast(y, model = "A?A")
#' }
#' @rdname ETSforecast
#' @export
ETSforecast = function(y, u = NULL, model = "???", s = frequency(y), h = max(2 * s, 6), criterion = "aicc", 
                    lambda = 1, armaIdent = FALSE, identAll = FALSE, forIntervals = FALSE,
                    bootstrap = FALSE, nSimul = 5000, verbose = FALSE,
                    alphaL = c(1e-8, 1-1e-8), betaL = alphaL, gammaL = alphaL, 
                    phiL = c(0.8, 0.98), p0 = -99999){
    m1 = ETSsetup(y, u, model, s, h, criterion, lambda, armaIdent, identAll, forIntervals,
                  bootstrap, nSimul, verbose, alphaL, betaL, gammaL, phiL, p0)
    m1 = ETSestim(m1)
    if (verbose)
            cat(m1$table)
    return(m1)
}
#' @title ETS
#' @description Runs all relevant functions for ETS modelling
#'
#' @details See help of \code{ETSforecast}.
#'
#' @param y a time series to forecast (it may be either a numerical vector or
#' a time series object). This is the only input required. If a vector, the additional
#' input \code{s} should be supplied compulsorily (see below).
#' @param u a matrix of input time series. If 
#' the output wanted to be forecast, matrix \code{u} should contain future values for inputs.
#' @param model the model to estimate. It is a single string indicating the type of 
#' model for each component with one or two letters:
#' \itemize{
#' \item Error: ? / A / M
#' 
#' \item Trend: ? / N / A / Ad / M / Md 
#' 
#' \item Seasonal: ? / N / A / M
#' 
#' }
#' 
#' @param s seasonal period of time series (1 for annual, 4 for quarterly, ...)
#' @param h forecast horizon. If the model includes inputs h is not used, the lenght of u is used instead.
#' @param criterion information criterion for identification ("aic", "bic" or "aicc").
#' @param lambda Box-Cox lambda parameter (NULL: estimate)
#' @param armaIdent check for arma models for error component (TRUE / FALSE).
#' @param identAll run all models to identify the best one (TRUE / FALSE)
#' @param forIntervals estimate forecasting intervals (TRUE / FALSE)
#' @param bootstrap use bootstrap simulation for predictive distributions
#' @param nSimul number of simulation runs for bootstrap simulation of predictive distributions
#' @param verbose intermediate estimation output (TRUE / FALSE)
#' @param alphaL constraints limits for alpha parameter
#' @param betaL constraints limits for beta parameter
#' @param gammaL constraints limits for gamma parameter
#' @param phiL constraints limits for phi parameter
#' @param p0 initial values for parameter search (alpha, beta, phi, gamma) with consraints:
#' \itemize{ 
#' \item 0 < alpha < 1
#' 
#' \item 0 < beta < alpha
#' 
#' \item 0 < phi < 1
#' 
#' \item 0 < gamma < 1 - alpha
#' }
#' 
#' @return An object of class \code{ETS}. It is a list with fields including all the inputs and
#'         the fields listed below as outputs. All the functions in this package fill in
#'         part of the fields of any \code{ETS} object as specified in what follows (function 
#'         \code{ETS} fills in all of them at once):
#' 
#' After running \code{ETSforecast}:
#' \item{p}{Estimated parameters}
#' \item{criteria}{Values for estimation criteria (LogLik, AIC, BIC, AICc)}
#' \item{yFor}{Forecasted values of output}
#' \item{yForV}{Variance of forecasted values of output}
#' \item{ySimul}{Bootstrap simulations for forecasting distribution evaluation}
#' 
#' After running \code{ETSvalidate}:
#' \item{table}{Estimation and validation table}
#' \item{comp}{Estimated components in matrix form}
#' 
#' After running \code{ETScomponents}:
#' \item{comp}{Estimated components in matrix form}
#' 
#' @author Diego J. Pedregal
#' 
#' @return An object of class \code{ETS}. See \code{ETSforecast}.
#' 
#' @seealso \code{\link{ETSforecast}}, \code{\link{ETSvalidate}},
#'          \code{\link{ETScomponents}}
#'          
#' @examples
#' \donttest{
#' y <- log(AirPassengers)
#' m1 <- ETS(y)
#' m1 <- ETS(y, model = "MAM")
#' }
#' @rdname ETS
#' @export
ETS = function(y, u = NULL, model = "???", s = frequency(y), h = 2 * s, criterion = "aicc", 
               lambda = 1, armaIdent = FALSE, identAll = FALSE, forIntervals = FALSE,
               bootstrap = FALSE, nSimul = 5000, verbose = FALSE,
               alphaL = c(1e-8, 1-1e-8), betaL = alphaL, gammaL = alphaL, 
               phiL = c(0.8, 0.98), p0 = -99999){
    m1 = ETSsetup(y, u, model, s, h, criterion, lambda, armaIdent, identAll, forIntervals,
                  bootstrap, nSimul, verbose, alphaL, betaL, gammaL, phiL, p0)
    m1 = ETSvalidate(m1)
    m1$v = m1$comp[, 1]
    if (verbose)
            cat(m1$table)
    return(m1)
}
