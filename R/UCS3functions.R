#' @title print.UComp
#' @description Prints an UComp object
#'
#' @details See help of \code{UC}.
#'
#' @param x Object of class \dQuote{UComp}.
#' @param ... Additional inputs to handle the way to print output.
#' 
#' @return No return value, called for side effects
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{UC}}, \code{\link{UCforecast}}, \code{\link{UCvalidate}}, \code{\link{UCfilter}}, \code{\link{UCsmooth}}, 
#'          \code{\link{UCdisturb}}, \code{\link{UCcomponents}}
#'          
#' @examples
#' \donttest{
#' y <- log(AirPassengers)
#' m1 <- UCforecast(y, model = "llt/equal/arma(0,0)")
#' print(m1)
#' }
#' @noRd
#' @export 
print.UComp = function(x, ...){
    if (x$model != "error") {
        x = UCvalidate(x, TRUE)
    } else {
        stop("ERROR: Model is not valid!!")
        return()
    }
}
#' @title summary.UComp
#' @description Prints an UComp object on screen
#'
#' @param object Object of class \dQuote{UComp}.
#' @param ... Additional inputs to function.
#' 
#' @details See help of \code{UC}.
#'
#' @return No return value, called for side effects
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{UC}}, \code{\link{UCforecast}}, \code{\link{UCvalidate}}, \code{\link{UCfilter}}, \code{\link{UCsmooth}}, 
#'          \code{\link{UCdisturb}}, \code{\link{UCcomponents}}
#'          
#' @examples
#' \donttest{
#' y <- log(AirPassengers)
#' m1 <- UCforecast(y, model = "llt/equal/arma(0,0)")
#' summary(m1)
#' }
#' @noRd
#' @export 
summary.UComp = function(object, ...){
    print(object)
}
#' @title plot.UComp
#' @description Plot components of UComp object
#'
#' @details See help of \code{UC}.
#'
#' @param x Object of class \dQuote{UComp}.
#' @param ... Additional inputs to function.
#' 
#' @return No return value, called for side effects
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{UC}}, \code{\link{UCforecast}}, \code{\link{UCvalidate}}, \code{\link{UCfilter}}, \code{\link{UCsmooth}}, 
#'          \code{\link{UCdisturb}}, \code{\link{UCcomponents}}
#'          
#' @examples
#' \donttest{
#' y <- log(AirPassengers)
#' m1 <- UCforecast(y, model = "llt/equal/arma(0,0)")
#' plot(m1)
#' }
#' @noRd
#' @export 
plot.UComp = function(x, ...){
    if (x$model == "error") {
        stop("ERROR: Model is not valid!!")
        return()
    }
    if (length(x$comp) < 2){
        VERBOSE = x$verbose
        x$verbose = FALSE
        x = UCcomponents(x)
        x$verbose = VERBOSE
    }
    if (is.ts(x$comp)){
        plot(x$comp, main = "Time Series Decomposition")
    } else {
        plot(ts(x$comp, frequency = x$periods[1]),
             main = "Time Series Decomposition")
    }
}
#' @title fitted.UComp
#' @description Fitted output values of UComp object
#'
#' @details See help of \code{UC}.
#'
#' @param object Object of class \dQuote{UComp}.
#' @param ... Additional inputs to function.
#' 
#' @return Fitted values of a UC model
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{UC}}, \code{\link{UCforecast}}, \code{\link{UCvalidate}}, \code{\link{UCfilter}}, \code{\link{UCsmooth}}, 
#'          \code{\link{UCdisturb}}, \code{\link{UCcomponents}}
#'          
#' @examples
#' \donttest{
#' y <- log(AirPassengers)
#' m1 <- UCforecast(y, model = "llt/equal/arma(0,0)")
#' fitted(m1)
#' }
#' @noRd
#' @export 
fitted.UComp = function(object, ...){
       if (object$model == "error") {
           stop("ERROR: Model is not valid!!")
           return()
        }
        if (length(object$yFit) < 2){
                VERBOSE = x$verbose
                x$verbose = FALSE
                object = UCsmooth(object)
                x$verbose = VERBOSE
        }
        return(object$yFit)
}
#' @title residuals.UComp
#' @description Standardised innovations values of UComp object
#'
#' @details See help of \code{UC}.
#'
#' @param object Object of class \dQuote{UComp}.
#' @param ... Additional inputs to function.
#' 
#' @return Residuals of a UC model
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{UC}}, \code{\link{UCforecast}}, \code{\link{UCvalidate}}, \code{\link{UCfilter}}, \code{\link{UCsmooth}}, 
#'          \code{\link{UCdisturb}}, \code{\link{UCcomponents}}
#'          
#' @examples
#' \donttest{
#' y <- log(AirPassengers)
#' m1 <- UCforecast(y, model = "llt/equal/arma(0,0)")
#' residuals(m1)
#' }
#' @noRd
#' @export 
residuals.UComp = function(object, ...){
    if (object$model == "error") {
        stop("ERROR: Model is not valid!!")
        return()
    }
    if (length(object$yFit) < 2){
        VERBOSE = object$verbose
        object$verbose = FALSE
        object = UCfilter(object)
        object$verbose = VERBOSE
    }
    return(object$v)
}
#' @title logLik.UComp
#' @description Extract log Likelihood value of UComp object
#'
#' @details See help of \code{UC}.
#'
#' @param object Object of class \dQuote{UComp}.
#' @param ... Additional inputs to function.
#' 
#' @return Log-Likelihood value of a UC model
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{UC}}, \code{\link{UCforecast}}, \code{\link{UCvalidate}}, \code{\link{UCfilter}}, \code{\link{UCsmooth}}, 
#'          \code{\link{UCdisturb}}, \code{\link{UCcomponents}}
#'          
#' @examples
#' \donttest{
#' y <- log(AirPassengers)
#' m1 <- UCforecast(y, model = "llt/equal/arma(0,0)")
#' logLik(m1)
#' }
#' @noRd
#' @export 
logLik.UComp = function(object, ...){
    if (object$model == "error") {
        stop("ERROR: Model is not valid!!")
        return()
    }
    out = object$criteria[1]
    class(out) = "logLik"
    attr(out, "df") = length(object$p) - 1
    attr(out, "nobs") = length(object$y)
    return(out)
}
#' @title AIC.UComp
#' @description Extract AIC value of UComp object
#'
#' @details Selection criteria for models with different number of 
#' parameters, the smaller AIC the better. The formula used here is
#' \eqn{AIC=-2 (ln(L) - k) / n}, where \eqn{ln(L)} is the log-likelihood
#' at the optimum, \eqn{k} is the number of parameters plus
#' non-stationary states and \eqn{n} is the number of observations.
#' Mind that this formulation differs from the usual definition that
#' does not divide by \eqn{n}. This makes that AIC(m) and AIC(logLik(m))
#' give different results, being m an UComp object.
#'
#' @param object Object of class \dQuote{UComp}.
#' @param ... Additional inputs to function.
#' @param k The penalty per parameter to be used.
#' 
#' @return AIC value of a UC model
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{UC}}, \code{\link{UCforecast}}, \code{\link{UCvalidate}}, \code{\link{UCfilter}}, \code{\link{UCsmooth}}, 
#'          \code{\link{UCdisturb}}, \code{\link{UCcomponents}}
#'          
#' @examples
#' \donttest{
#' y <- log(AirPassengers)
#' m1 <- UCforecast(y, model = "llt/equal/arma(0,0)")
#' AIC(m1)
#' }
#' @export 
AIC.UComp = function(object, ..., k = 2){
     if (object$model == "error") {
         stop("ERROR: Model is not valid!!")
        return()
     }
     return(object$criteria[2])
}
#' @title BIC.UComp
#' @description Extract BIC (or SBC) value of UComp object
#'
#' @details Selection criteria for models with different number of 
#' parameters, the smaller BIC the better. The formula used here is
#' \eqn{BIC=(-2 ln(L) + k ln(n)) / n}, where \eqn{ln(L)} is the log-likelihood
#' at the optimum, \eqn{k} is the number of parameters plus
#' non-stationary states and \eqn{n} is the number of observations.
#' Mind that this formulation differs from the usual definition that
#' does not divide by \eqn{n}. This makes that BIC(m) and BIC(logLik(m))
#' give different results, being m an UComp object.
#'
#' @param object Object of class \dQuote{UComp}.
#' @param ... Additional inputs to function.
#' 
#' @return BIC value of a UC model
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{UC}}, \code{\link{UCforecast}}, \code{\link{UCvalidate}}, \code{\link{UCfilter}}, \code{\link{UCsmooth}}, 
#'          \code{\link{UCdisturb}}, \code{\link{UCcomponents}}
#'          
#' @examples
#' \donttest{
#' y <- log(AirPassengers)
#' m1 <- UCforecast(y, model = "llt/equal/arma(0,0)")
#' BIC(m1)
#' }
#' @export 
BIC.UComp = function(object, ...){
    if (object$model == "error") {
        stop("ERROR: Model is not valid!!")
        return()
    }
    return(object$criteria[3])
}
#' @title coef.UComp
#' @description Extracts model coefficients of UComp object
#'
#' @details See help of \code{UC}.
#'
#' @param object Object of class \dQuote{UComp}.
#' @param ... Ignored.
#' 
#' @return Estimated coefficients of a UC model
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{UC}}, \code{\link{UCforecast}}, \code{\link{UCvalidate}}, \code{\link{UCfilter}}, \code{\link{UCsmooth}}, 
#'          \code{\link{UCdisturb}}, \code{\link{UCcomponents}}
#'          
#' @examples
#' \donttest{
#' y <- log(AirPassengers)
#' m1 <- UCforecast(y, model = "llt/equal/arma(0,0)")
#' coef(m1)
#' }
#' @noRd
#' @export 
coef.UComp = function(object, ...){
    if (object$model == "error") {
        stop("ERROR: Model is not valid!!")
        return()
    }
    if (length(object$table) < 2){
        VERBOSE = object$verbose
        object$verbose = FALSE
        object = UCvalidate(object, FALSE)
        object$verbose = VERBOSE
    }
    return(object$p)
}
#' @title predict.UComp
#' @description Forecasting using structural Unobseved Components models with prediction intervals
#'
#' @details See help of \code{UC}.
#'
#' @param object Object of class \dQuote{UComp}.
#' @param newdata New output data to apply \dQuote{UComp} object to.
#' @param n.ahead Number of steps ahead to forecast or new inputs variables 
#' including their predictions.
#' @param level Confidence level for prediction intervals.
#' @param ... Ignored.
#' 
#' @return Forecasts of a UC model
#' 
#' @author Diego J. Pedregal
#' 
#' @return A matrix with the mean forecasts and lower and upper prediction intervals
#' 
#' @seealso \code{\link{UC}}, \code{\link{UCforecast}}, \code{\link{UCvalidate}}, \code{\link{UCfilter}}, \code{\link{UCsmooth}}, 
#'          \code{\link{UCdisturb}}, \code{\link{UCcomponents}}
#'          
#' @examples
#' \donttest{
#' y <- log(AirPassengers)
#' m1 <- UCforecast(y, model = "llt/eq/arma(0,0)")
#' f1 <- predict(m1)
#' }
#' @export 
predict.UComp = function(object, newdata = NULL, n.ahead = NULL, level = 0.95, ...){
    if (object$model == "error") {
        stop("ERROR: Model is not valid!!")
        return()
    }
    cnst = qt(level + (1 - level) / 2, length(object$y) - length(object$p))
    if (!is.null(newdata)){
        object$y = newdata
    }
   if (!is.null(n.ahead) && length(size(n.ahead)) == 1){
        object$h = n.ahead
    }
    if (!is.null(n.ahead) && length(size(n.ahead)) > 1){
        object$u = n.ahead
    }
    m = UCfilter(object)
    pred = as.numeric(tail(m$yFit, m$h))
    predS = as.numeric(sqrt(tail(m$yFitV, m$h)))
    out = cbind(pred, pred - cnst * predS, pred + cnst * predS)
    if (is.ts(object$y)){
        aux = ts(matrix(0, length(object$y) + 1), start = start(object$y),
                 frequency = frequency(object$y))
        stDate = end(aux)
        freq = frequency(object$y)
        out = ts(out, start = stDate, frequency = freq)
    }
    colnames(out) = c("frcst", "lower", "upper")
    return(out)
}
#' @title tsdiag.UComp
#' @description Diagnostic plots for UComp objects
#'
#' @details See help of \code{UC}.
#'
#' @param object Object of class \dQuote{UComp}.
#' @param gof.lag Maximum number of lags for pormanteau Ljung-Box test
#' @param ... Additional inputs to function.
#' 
#' @return Diagnostics of a UC model
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{UC}}, \code{\link{UCforecast}}, \code{\link{UCvalidate}}, \code{\link{UCfilter}}, \code{\link{UCsmooth}}, 
#'          \code{\link{UCdisturb}}, \code{\link{UCcomponents}}
#'          
#' @examples
#' \donttest{
#' y <- log(AirPassengers)
#' m1 <- UCforecast(y, model = "llt/eq/arma(0,0)")
#' tsdiag(m1)
#' }
#' @noRd
#' @export 
tsdiag.UComp = function(object, gof.lag = NULL, ...){
    if (object$model == "error") {
        stop("ERROR: Model is not valid!!")
        return()
    }
    if (length(object$v) < 2){
        object = UCfilter(object)
    }
    # Number of u's
    nu = dim(object$u)[1]
    if (dim(object$u)[2] == 2){
        nu = 0
    }
    # Number of parameters
    nPar = length(object$p) - 1 + nu
    aux = list(residuals = object$v,
               sigma2 = 1,
               nobs = length(object$y) - nPar,
               coef = object$p,
               x = object$y,
               fitted = object$yFit)
    if (is.null(gof.lag)){
        tsdiag(structure(aux, class = "Arima"))
    } else {
        tsdiag(structure(aux, class = "Arima"), gof.lag)
    }
}
#' @title getp0
#' @description Get initial conditions for parameters of \code{UComp} object
#'
#' @details Provides initial parameters of a given model for the time series.
#' They may be changed arbitrarily by the user to include as an input \code{p0} to
#' \code{UC} or \code{UCforecast} functions (see example below).
#' There is no guarantee that the model will converge and selecting initial conditions
#' should be used with care.
#'
#' @param y a time series to forecast.
#' @param model any valid \code{UComp} model without any ?.
#' @param periods vector of fundamental period and harmonics required.
#' 
#' @author Diego J. Pedregal
#' 
#' @return A set of parameters p0 of an object of class \code{UComp}
#' to use as input to \code{\link{UC}}, \code{\link{UCforecast}}.
#' 
#' @seealso \code{\link{UC}}, \code{\link{UCvalidate}}, \code{\link{UCfilter}}, \code{\link{UCsmooth}}, 
#'          \code{\link{UCdisturb}}, \code{\link{UCcomponents}},
#'          \code{\link{UChp}}
#'          
#' @examples
#' \dontrun{
#' p0 <- getp0(log(AirPassengers), model = "llt/equal/arma(0,0)")
#' p0[1] <- 0  # p0[1] <- NA
#' m <- UCforecast(log(AirPassengers), model = "llt/equal/arma(0,0)", p0 = p0)
#' }
#' @rdname getp0
#' @export
getp0 = function(y, model = "llt/equal/arma(0,0)", periods = NA){
    if (model == "error") {
        stop("ERROR: Model is not valid!!")
        return()
    }
    if (any(utf8ToInt(model) == utf8ToInt("?"))){
        stop("UComp ERROR: Model should not contain any \'?\'!!!")
    }
    sys = UCsetup(y, model = model, periods = periods, verbose = FALSE)
    sys = UCestim(sys)
    p0 = as.vector(sys$p0)
    p1 = coef(sys)
    names(p0) = names(p1)
    
    return(p0)
}

