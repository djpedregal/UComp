#' @title print.ARIMA
#' @description Prints an ARIMA object
#'
#' @details See help of \code{ARIMA}.
#'
#' @param x Object of class \dQuote{ARIMA}.
#' @param ... Additional inputs to handle the way to print output.
#' 
#' @return No return value, called for side effects
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{ARIMA}}, \code{\link{ARIMAforecast}}, \code{\link{ARIMAvalidate}},
#'          
#' @examples
#' \donttest{
#' m1 <- ARIMAforecast(log(gdp))
#' print(m1)
#' }
#' @rdname print
#' @noRd
#' @export 
print.ARIMA = function(x, ...){
    if (length(x$table) < 3) {
        x = ARIMAvalidate(x)
    }
    cat(x$table)
}
#' @title summary.ARIMA
#' @description Prints an ARIMA object on screen
#'
#' @param object Object of class \dQuote{ARIMA}.
#' @param ... Additional inputs to function.
#' 
#' @details See help of \code{ARIMA}.
#'
#' @return No return value, called for side effects
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{ARIMA}}, \code{\link{ARIMAforecast}}, \code{\link{ARIMAvalidate}},
#'          
#' @examples
#' \donttest{
#' m1 <- ARIMAforecast(log(gdp))
#' summary(m1)
#' }
#' @rdname summary.ARIMA
#' @noRd
#' @export 
summary.ARIMA = function(object, ...){
    print(object)
}
#' @title residuals.ARIMA
#' @description Residuals values of ARIMA object
#'
#' @details See help of \code{ARIMA}.
#'
#' @param object Object of class \dQuote{ARIMA}.
#' @param ... Additional inputs to function.
#' 
#' @return Error from ARIMA model
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{ARIMA}}, \code{\link{ARIMAforecast}}, \code{\link{ARIMAvalidate}},
#'          
#' @examples
#' \donttest{
#' y <- log(AirPassengers)
#' m1 <- ARIMA(y)
#' residuals(m1)
#' }
#' @noRd
#' @export 
residuals.ARIMA = function(object, ...){
    return(object$error)
}
#' @title plot.ARIMA
#' @description Plot zplane of ARIMA object
#'
#' @details See help of \code{ARIMA}.
#'
#' @param x Object of class \dQuote{ARIMA}.
#' @param ... Additional inputs to function.
#' 
#' @return No return value, called for side effects
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{ARIMA}}, \code{\link{ARIMAforecast}}, \code{\link{ARIMAvalidate}},
#'          
#' @examples
#' \donttest{
#' m1 <- ARIMAforecast(log(gdp))
#' plot(m1)
#' }
#' @rdname plot.ARIMA
#' @noRd
#' @export 
plot.ARIMA = function(x, ...){
   if (is.null(x$p)){
           stop("ERROR: Please, estimate the model first!!!")
   } else {
        ARaux = matrix(1, x$model[1] + 1)
        ARS = matrix(0, x$model[4] * x$s + 1)
        MAaux = matrix(1, x$model[3] + 1)
        MAS = matrix(0, x$model[6] * x$s + 1)
        ARaux[1] = 1
        MAaux[1] = 1
        ARS[1] = 1
        MAS[1] = 1
        i = 1
        if (x$model[1] > 0){
                for (j in 1 : x$model[1]){
                        ARaux[j + 1] = x$p[i]
                        i = i + 1
                }
        }
        if (x$model[4] > 0){
                for (j in 1 : x$model[4]){
                        ARS[j * x$s + 1] = x$p[i]
                        i = i + 1
                }
        }
        if (x$model[3] > 0){
                for (j in 1 : x$model[3]){
                        MAaux[j + 1] = x$p[i]
                        i = i + 1
                }
        }
        if (x$model[6] > 0){
                for (j in 1 : x$model[6]){
                        MAS[j * x$s + 1] = x$p[i]
                        i = i + 1
                }
        }
        AR = conv(ARaux, ARS)
        MA = conv(MAaux, MAS)
        zplane(MA, AR)
   }
}
