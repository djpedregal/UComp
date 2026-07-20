#' @title print.ETS
#' @description Prints an ETS object
#'
#' @details See help of \code{ETS}.
#'
#' @param x Object of class \dQuote{ETS}.
#' @param ... Additional inputs to handle the way to print output.
#' 
#' @return No return value, called for side effects
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{ETS}}, \code{\link{ETSforecast}}, \code{\link{ETSvalidate}},
#'          \code{\link{ETScomponents}}, \code{\link{ETSestim}}
#'          
#' @examples
#' \donttest{
#' m1 <- ETSforecast(log(gdp))
#' print(m1)
#' }
#' @rdname print
#' @noRd
#' @export 
print.ETS = function(x, ...){
    if (length(x$table) < 3){
        x = ETSvalidate(x)
    }
    cat(x$table)
}
#' @title summary.ETS
#' @description Prints an ETS object on screen
#'
#' @param object Object of class \dQuote{ETS}.
#' @param ... Additional inputs to function.
#' 
#' @details See help of \code{ETS}.
#'
#' @return No return value, called for side effects
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{ETS}}, \code{\link{ETSforecast}}, \code{\link{ETSvalidate}},
#'          \code{\link{ETScomponents}}, \code{\link{ETSestim}}
#'          
#' @examples
#' \donttest{
#' m1 <- ETSforecast(log(gdp))
#' summary(m1)
#' }
#' @rdname summary.ETS
#' @noRd
#' @export 
summary.ETS = function(object, ...){
    print(object)
}
#' @title plot.ETS
#' @description Plot components of ETS object
#'
#' @details See help of \code{ETS}.
#'
#' @param x Object of class \dQuote{ETS}.
#' @param ... Additional inputs to function.
#' 
#' @return No return value, called for side effects
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{ETS}}, \code{\link{ETSforecast}}, \code{\link{ETSvalidate}},
#'          \code{\link{ETScomponents}}, \code{\link{ETSestim}}
#'          
#' @examples
#' \donttest{
#' m1 <- ETSforecast(log(gdp))
#' plot(m1)
#' }
#' @rdname plot
#' @noRd
#' @export 
plot.ETS = function(x, ...){
    if (length(x$comp) < 2){
        x = ETScomponents(x)
    }
    if (is.ts(x$comp)){
        plot(x$comp, main = "Time Series Decomposition")
    } else {
        plot(ts(x$comp, frequency = x$s),
             main = "Time Series Decomposition")
    }
}
#' @title fitted.ETS
#' @description Fitted output values of ETS object
#'
#' @details See help of \code{ETS}.
#'
#' @param object Object of class \dQuote{ETS}.
#' @param ... Additional inputs to function.
#' 
#' @return Fitted values from ETS model
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{ETS}}, \code{\link{ETSforecast}}, \code{\link{ETSvalidate}},
#'          \code{\link{ETScomponents}}, \code{\link{ETSestim}}
#'          
#' @examples
#' \donttest{
#' m1 <- ETSforecast(log(gdp))
#' fitted(m1)
#' }
#' @rdname fitted
#' @noRd
#' @export 
fitted.ETS = function(object, ...){
    if (length(object$comp) < 2){
        object = ETScomponents(object)
    }
    return(object$com[, 2])
}
#' @title residuals.ETS
#' @description Residuals of ETS object
#'
#' @details See help of \code{ETS}.
#'
#' @param object Object of class \dQuote{ETS}.
#' @param ... Additional inputs to function.
#' 
#' @return Residuals of ETS model
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{ETS}}, \code{\link{ETSforecast}}, \code{\link{ETSvalidate}},
#'          \code{\link{ETScomponents}}, \code{\link{ETSestim}}
#'          
#' @examples
#' \donttest{
#' m1 <- ETSforecast(log(gdp))
#' residuals(m1)
#' }
#' @rdname residuals
#' @noRd
#' @export 
residuals.ETS = function(object, ...){
    if (length(object$comp) < 2){
        object = ETScomponents(object)
    }
    return(object$com[, 1])
}
#' @title auxInvBoxCox
#' @description Inverse of Box-Cox transformation
#'
#' @param y matrix, array or vector
#' @param lambda lambda parameter of Box-Cox transformation
#' 
#' @return Inverse of Box-Cox heteroskedasticity transformation
#' 
#' @author Diego J. Pedregal
#' 
#' @rdname auxInvBoxCox
#' @export
auxInvBoxCox = function(y, lambda){
 if (abs(lambda) < 0.02){
  out = exp(y)
 } else if (lambda > 0.98) {
  out = y
 } else {
  out = (y * lambda + 1) ^ (1 / lambda)
 }
 return(out)
}
#' @title invBoxCox
#' @description Calculates inverse of Box-Cox transformation 
#' with confidence bands, calculated as const time the standard error
#'
#' @param y matrix, array or vector
#' @param yVar matrix, array or vector of variances of y
#' @param lambda lambda parameter of Box-Cox transformation
#' @param const number of standard error for confidence band
#' 
#' @return Inverse of Box-Cox heteroskedasticity transformation with confidence bands
#' 
#' @author Diego J. Pedregal
#' 
#' @rdname invBoxCox
#' @export
invBoxCox = function(y, yVar, lambda, const = 2){
 if (is.vector(y)){
  out = cbind(auxInvBoxCox(y, lambda), auxInvBoxCox(y - const * sqrt(yVar), lambda), 
              auxInvBoxCox(y + const * sqrt(yVar), lambda))
  colnames(out) = c("y", "conf-", "conf+")
 } else {
  out = list(y = auxInvBoxCox(y, lambda), "conf-" = auxInvBoxCox(y - const * sqrt(yVar), lambda), 
             "conf+" = auxInvBoxCox(y + const * sqrt(yVar), lambda))
 }
 return(out)
}

