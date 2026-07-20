#' @title print.TETS
#' @description Prints a TOBIT TETS object
#'
#' @details See help of \code{TETS}.
#'
#' @param x Object of class \dQuote{TETS}.
#' @param ... Additional inputs to handle the way to print output.
#' 
#' @return No return value, called for side effects
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{TETS}}, \code{\link{TETSforecast}}, \code{\link{TETSvalidate}},
#'          \code{\link{TETScomponents}}, \code{\link{TETSestim}}
#'          
#' @examples
#' \donttest{
#' m1 <- TETSforecast(log(gdp))
#' print(m1)
#' }
#' @rdname print
#' @noRd
#' @export 
print.TETS = function(x, ...){
    if (length(x$table) < 3){
        x = TETSvalidate(x)
    }
    cat(x$table)
}
#' @title summary.TETS
#' @description Prints a TOBIT TETS object on screen
#'
#' @param object Object of class \dQuote{TETS}.
#' @param ... Additional inputs to function.
#' 
#' @details See help of \code{TETS}.
#'
#' @return No return value, called for side effects
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{TETS}}, \code{\link{TETSforecast}}, \code{\link{TETSvalidate}},
#'          \code{\link{TETScomponents}}, \code{\link{TETSestim}}
#'          
#' @examples
#' \donttest{
#' m1 <- TETSforecast(log(gdp))
#' summary(m1)
#' }
#' @rdname summary.TETS
#' @noRd
#' @export 
summary.TETS = function(object, ...){
    print(object)
}
#' @title plot.TETS
#' @description Plot components of TETS object
#'
#' @details See help of \code{TETS}.
#'
#' @param x Object of class \dQuote{TETS}.
#' @param ... Additional inputs to function.
#' 
#' @return No return value, called for side effects
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{TETS}}, \code{\link{TETSforecast}}, \code{\link{TETSvalidate}},
#'          \code{\link{TETScomponents}}, \code{\link{TETSestim}}
#'          
#' @examples
#' \donttest{
#' m1 <- TETSforecast(log(gdp))
#' plot(m1)
#' }
#' @rdname plot
#' @noRd
#' @export 
plot.TETS = function(x, ...){
    if (length(x$comp) < 2){
        x = TETScomponents(x)
    }
    if (is.ts(x$comp)){
        plot(x$comp, main = "Time Series Decomposition")
    } else {
        plot(ts(x$comp, frequency = x$s),
             main = "Time Series Decomposition")
    }
}
#' @title fitted.TETS
#' @description Fitted output values of TETS object
#'
#' @details See help of \code{TETS}.
#'
#' @param object Object of class \dQuote{TETS}.
#' @param ... Additional inputs to function.
#' 
#' @return Fitted values of TETS model
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{TETS}}, \code{\link{TETSforecast}}, \code{\link{TETSvalidate}},
#'          \code{\link{TETScomponents}}, \code{\link{TETSestim}}
#'          
#' @examples
#' \donttest{
#' m1 <- TETSforecast(log(gdp))
#' fitted(m1)
#' }
#' @rdname fitted
#' @noRd
#' @export 
fitted.TETS = function(object, ...){
    if (length(object$comp) < 2){
        object = TETScomponents(object)
    }
    return(object$com[, 2])
}
#' @title residuals.TETS
#' @description Residuals of TETS object
#'
#' @details See help of \code{TETS}.
#'
#' @param object Object of class \dQuote{TETS}.
#' @param ... Additional inputs to function.
#' 
#' @return Residuals of TETS model
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{TETS}}, \code{\link{TETSforecast}}, \code{\link{TETSvalidate}},
#'          \code{\link{TETScomponents}}, \code{\link{TETSestim}}
#'          
#' @examples
#' \donttest{
#' m1 <- TETSforecast(log(gdp))
#' residuals(m1)
#' }
#' @rdname residuals
#' @noRd
#' @export 
residuals.TETS = function(object, ...){
    if (length(object$comp) < 2){
        object = TETScomponents(object)
    }
    return(object$com[, 1])
}
