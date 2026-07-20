#' @title ARIMAsetup
#' @description Sets up ARIMA general models
#'
#' @details See help of \code{ARIMAforecast}.
#'
#' @param y a time series to forecast (it may be either a numerical vector or
#' a time series object). This is the only input required. If a vector, the additional
#' input \code{s} should be supplied compulsorily (see below).
#' @param u a matrix of input time series. If 
#' the output wanted to be forecast, matrix \code{u} should contain future values for inputs.
#' @param model the model to estimate. A vector c(p,d,q,P,D,Q) containing the model orders
#'               of an ARIMA(p,d,q)x(P,D,Q)_s model. A constant may be estimated with the
#'               cnst input.
#'               Use a NULL to automatically identify the ARIMA model.
#' @param cnst flag to include a constant in the model (TRUE/FALSE/NULL). Use NULL to estimate
#' @param s seasonal period of time series (1 for annual, 4 for quarterly, ...)
#' @param h forecast horizon. If the model includes inputs h is not used, the lenght of u is used instead.
#' @param criterion information criterion for identification stage ("aic", "bic", "aicc")
#' @param verbose intermediate estimation output (TRUE / FALSE)
#' @param lambda Box-Cox lambda parameter (NULL: estimate)
#' @param maxOrders a vector c(p,d,q,P,D,Q) containing the maximum orders of model orders 
#'                  to search for in the automatic identification
#' @param bootstrap use bootstrap simulation for predictive distributions
#' @param nSimul number of simulation runs for bootstrap simulation of predictive distributions
#' @param fast fast identification (avoids post-identification checks)
#' 
#' @author Diego J. Pedregal
#' 
#' @return An object of class \code{ARIMA}. It is a list with fields including all the inputs and
#'         the fields listed below as outputs. All the functions in this package fill in
#'         part of the fields of any \code{ARIMA} object as specified in what follows (function 
#'         \code{ARIMA} fills in all of them at once):
#' 
#' After running \code{ARIMAforecast} or \code{ARIMA}:
#' \item{p}{Estimated parameters}
#' \item{yFor}{Forecasted values of output}
#' \item{yForV}{Variance of forecasted values of output}
#' \item{ySimul}{Bootstrap simulations for forecasting distribution evaluation}
#' 
#' After running \code{ARIMAvalidate}:
#' \item{table}{Estimation and validation table}
#' 
#' @seealso \code{\link{ARIMA}}, \code{\link{ARIMAforecast}}, \code{\link{ARIMAvalidate}},
#'          
#' @examples
#' \donttest{
#' y <- log(AirPassengers)
#' m1 <- ARIMAsetup(y)
#' m1 <- ARIMAsetup(y, lambda = NULL)
#' }
#' @rdname ARIMAsetup
#' @export
ARIMAsetup = function(y, u = NULL, model = NULL, cnst = NULL, s = frequency(y), 
                      criterion = "bic", h = 2 * s, verbose = FALSE, lambda = 1, 
                      maxOrders = c(3, 2, 3, 2, 1, 2), bootstrap = FALSE, nSimul = 5000,
                      fast = FALSE){
        y = as.ts(y)
        if (s > 24)
                stop("Error: Data with period greater than 24 are not allowed!!")
        if (length(y) < s + 2 || length(y) < 8)
                stop("Error: Not enough data to estimate model!!")
        if (is.null(lambda))
                lambda = 9999.9
        if (typeof(cnst) == "logical" && cnst){
                cnst = 1.0
        } else if (typeof(cnst) == "logical" && !cnst){
                cnst = 0.0
        } else if (is.null(cnst)){
                cnst = 9999.9
        }
        out =  list(y = y,
                    u = u,
                    model = model,
                    cnst = cnst,
                    s = s,
                    h = h,
                    lambda = lambda,
                    bootstrap = bootstrap,
                    nSimul = nSimul,
                    verbose = verbose,
                    maxOrders = maxOrders,
                    criterion = criterion,
                    fast = fast,
                    identDiff = TRUE,
                    identMethod = "gm",
                    # outputs
                    error = NULL,
                    yFor = NULL,
                    yForV = NULL,
                    ySimul = NULL,
                    table = "",
                    p = NULL,
                    BIC = NA,
                    AIC = NA,
                    AICc = NA,
                    IC = NA,
                    v = NULL)
        return(structure(out, class = "ARIMA"))
}
#' @title ARIMAforecast
#' @description Estimates and forecasts ARIMA general univariate models
#'
#' @details \code{ARIMAforecast} is a function for modelling and forecasting univariate
#' time series with Autoregressive Integrated Moving Average (ARIMA) time series models. 
#' It sets up the model with a number of control variables that
#' govern the way the rest of functions in the package will work. It also estimates 
#' the model parameters by Maximum Likelihood and forecasts the data.
#'
#' @inheritParams ARIMA
#' 
#' @return An object of class \code{ARIMA}. It is a list with fields including all the inputs and
#'         the fields listed below as outputs. All the functions in this package fill in
#'         part of the fields of any \code{ARIMA} object as specified in what follows (function 
#'         \code{ARIMA} fills in all of them at once):
#' 
#' After running \code{ARIMAforecast} or \code{ARIMA}:
#' \item{p}{Estimated parameters}
#' \item{yFor}{Forecasted values of output}
#' \item{yForV}{Variance of forecasted values of output}
#' \item{ySimul}{Bootstrap simulations for forecasting distribution evaluation}
#' 
#' After running \code{ARIMAvalidate}:
#' \item{table}{Estimation and validation table}
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{ARIMA}}, \code{\link{ARIMAvalidate}},
#'          
#' @examples
#' \donttest{
#' y <- log(AirPassengers)
#' m1 <- ARIMAforecast(y)
#' m1 <- ARIMAforecast(y, lambda = NULL)
#' }
#' @rdname ARIMAforecast
#' @export
ARIMAforecast = function(y, u = NULL, model = NULL, cnst = NULL, s = frequency(y), 
                      criterion = "bic", h = 2 * s, verbose = FALSE, lambda = 1, 
                      maxOrders = c(3, 2, 3, 2, 1, 2), bootstrap = FALSE, nSimul = 5000,
                      fast = FALSE){
        m = ARIMAsetup(y, u, model, cnst, s, criterion, h, FALSE, lambda, maxOrders,
                       bootstrap, nSimul, fast)
        m = ARIMAestim(m)
        IC = m$IC
        if (is.null(IC) || !is.finite(IC)){
                IC = 1e10
        }
        if ((is.null(model) && !fast) || !is.finite(IC)){
                if (s == 1 && sum(abs(m$model[1 : 3] - c(0, 1, 1))) != 0){
                        # Yearly data
                        model = c(0, 1, 1, 0, 0, 0)
                        m1 = ARIMAsetup(y, u, model, cnst, s, criterion, h, FALSE, lambda, maxOrders,
                                        bootstrap, nSimul, fast)
                        m1 = ARIMAestim(m1)
                        if (is.finite(m1$IC) && m1$IC < IC) {
                                IC = m1$IC
                                m = m1
                        }
                        if (m$model[2] > 0){
                                model = m$model
                                model[2] = model[2] - 1
                                model[3] = min(model[3] + 1, maxOrders[3])
                                if (!all(model == c(0, 1, 1, 0, 0, 0))){
                                        m1 = ARIMAsetup(y, u, model, cnst, s, criterion, h, FALSE, lambda, maxOrders,
                                                        bootstrap, nSimul, fast)
                                        m1 = ARIMAestim(m1)
                                        if (is.finite(m1$IC) && m1$IC < IC){
                                                IC = m1$IC
                                                m = m1
                                        }
                                }
                        }
               } else if (s > 1){
                        # Non-yearly data
                        if (sum(abs(m$model - c(0, 1, 1, 0, 1, 1))) != 0){
                                model = c(0, 1, 1, 0, 1, 1)
                                m1 = ARIMAsetup(y, u, model, cnst, s, criterion, h, FALSE, lambda, maxOrders,
                                                bootstrap, nSimul, fast)
                                m1 = ARIMAestim(m1)
                                if (is.finite(m1$IC) && m1$IC < IC){
                                        IC = m1$IC
                                        m = m1
                                }
                        }
                        if (m$model[2] > 0 && m$model[5] > 0){
                                model = m$model
                                model[2] = model[2] - 1
                                model[1] = min(model[1] + 1, maxOrders[1])
                                if (!all(model == c(0, 1, 1, 0, 1, 1))){
                                        m1 = ARIMAsetup(y, u, model, cnst, s, criterion, h, FALSE, lambda, maxOrders,
                                                        bootstrap, nSimul, fast)
                                        m1 = ARIMAestim(m1)
                                        if (is.finite(m1$IC) && m1$IC < IC){
                                                IC = m1$IC
                                                m = m1
                                        }
                                }
                        }
                }
        }
        if (verbose)
                cat(m$table)
        return(m)
}
#' @title ARIMA
#' @description Runs all relevant functions for ARIMA modelling
#'
#' @details See help of \code{ARIMAforecast}.
#'
#' @param y a time series to forecast (it may be either a numerical vector or
#' a time series object). This is the only input required. If a vector, the additional
#' input \code{s} should be supplied compulsorily (see below).
#' @param u a matrix of input time series. If 
#' the output wanted to be forecast, matrix \code{u} should contain future values for inputs.
#' @param model the model to estimate. A vector c(p,d,q,P,D,Q) containing the model orders
#'               of an ARIMA(p,d,q)x(P,D,Q)_s model. A constant may be estimated with the
#'               cnst input.
#'               Use a NULL to automatically identify the ARIMA model.
#' @param cnst flag to include a constant in the model (TRUE/FALSE/NULL). Use NULL to estimate
#' @param s seasonal period of time series (1 for annual, 4 for quarterly, ...)
#' @param h forecast horizon. If the model includes inputs h is not used, the lenght of u is used instead.
#' @param criterion information criterion for identification stage ("aic", "bic", "aicc")
#' @param verbose intermediate estimation output (TRUE / FALSE)
#' @param lambda Box-Cox lambda parameter (NULL: estimate)
#' @param maxOrders a vector c(p,d,q,P,D,Q) containing the maximum orders of model orders 
#'                  to search for in the automatic identification
#' @param bootstrap use bootstrap simulation for predictive distributions
#' @param nSimul number of simulation runs for bootstrap simulation of predictive distributions
#' @param fast fast identification (avoids post-identification checks)
#' 
#' @author Diego J. Pedregal
#' 
#' @return An object of class \code{ARIMA}. It is a list with fields including all the inputs and
#'         the fields listed below as outputs. All the functions in this package fill in
#'         part of the fields of any \code{ARIMA} object as specified in what follows (function 
#'         \code{ARIMA} fills in all of them at once):
#' 
#' After running \code{ARIMAforecast} or \code{ARIMA}:
#' \item{p}{Estimated parameters}
#' \item{yFor}{Forecasted values of output}
#' \item{yForV}{Variance of forecasted values of output}
#' \item{ySimul}{Bootstrap simulations for forecasting distribution evaluation}
#' 
#' After running \code{ARIMAvalidate}:
#' \item{table}{Estimation and validation table}
#' 
#' @seealso \code{\link{ARIMAforecast}}, \code{\link{ARIMAvalidate}},
#'          
#' @examples
#' \donttest{
#' y <- log(AirPassengers)
#' m1 <- ARIMA(y)
#' m1 <- ARIMA(y, lambda = NULL)
#' }
#' @rdname ARIMA
#' @export
ARIMA = function(y, u = NULL, model = NULL, cnst = NULL, s = frequency(y), 
                 criterion = "bic", h = 2 * s, verbose = FALSE, lambda = 1, 
                 maxOrders = c(3, 2, 3, 2, 1, 2), bootstrap = FALSE, nSimul = 5000,
                 fast = FALSE){
        m = ARIMAsetup(y, u, model, cnst, s, criterion, h, FALSE, lambda, maxOrders,
                       bootstrap, nSimul, fast)
        m = ARIMAvalidate(m)
        IC = m$IC
        if (is.null(model) && !fast || !is.finite(IC)){
                if (s == 1 && sum(abs(m$model[1 : 3] - c(0, 1, 1))) != 0){
                        # Yearly data
                        model = c(0, 1, 1, 0, 0, 0)
                        m1 = ARIMAsetup(y, u, model, cnst, s, criterion, h, FALSE, lambda, maxOrders,
                                        bootstrap, nSimul, fast)
                        m1 = ARIMAvalidate(m1)
                        if (is.finite(m1$IC) && m1$IC < IC){
                                IC = m1$IC
                                m = m1
                        }
                        if (m$model[2] > 0){
                                model = m$model
                                model[2] = model[2] - 1
                                model[3] = min(model[3] + 1, maxOrders[3])
                                if (!all(model == c(0, 1, 1, 0, 0, 0))){
                                        m1 = ARIMAsetup(y, u, model, cnst, s, criterion, h, FALSE, lambda, maxOrders,
                                                        bootstrap, nSimul, fast)
                                        m1 = ARIMAvalidate(m1)
                                        if (is.finite(m1$IC) && m1$IC < IC){
                                                IC = m1$IC
                                                m = m1
                                        }
                                }
                        }
                } else if (s > 1){
                        # Non-yearly data
                        if (sum(abs(m$model - c(0, 1, 1, 0, 1, 1))) != 0){
                                model = c(0, 1, 1, 0, 1, 1)
                                m1 = ARIMAsetup(y, u, model, cnst, s, criterion, h, FALSE, lambda, maxOrders,
                                                bootstrap, nSimul, fast)
                                m1 = ARIMAvalidate(m1)
                                if (is.finite(m1$IC) && m1$IC < IC){
                                        IC = m1$IC
                                        m = m1
                                }
                        }
                        if (m$model[2] > 0 && m$model[5] > 0){
                                model = m$model
                                model[2] = model[2] - 1
                                model[1] = min(model[1] + 1, maxOrders[1])
                                if (!all(model == c(0, 1, 1, 0, 1, 1))){
                                        m1 = ARIMAsetup(y, u, model, cnst, s, criterion, h, FALSE, lambda, maxOrders,
                                                        bootstrap, nSimul, fast)
                                        m1 = ARIMAvalidate(m1)
                                        if (is.finite(m1$IC) && m1$IC < IC){
                                                IC = m1$IC
                                                m = m1
                                        }
                                }
                        }
                }
        }
        if (verbose)
                cat(m$table)
        return(m)
}
