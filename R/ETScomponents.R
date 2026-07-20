#' @title ETScomponents
#' @description Estimates components of ETS models
#'
#' @param m an object of type \code{ETS} created with \code{ETSforecast}
#' 
#' @return The same input object with the appropriate fields 
#' filled in, in particular:
#' \item{comp}{Estimated components in matrix form}
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{ETS}}, \code{\link{ETSforecast}}, \code{\link{ETSvalidate}}
#'        
#' @examples
#' \donttest{
#' m1 <- ETS(log(gdp))
#' m1 <- ETScomponents(m1)
#' }
#' @rdname ETScomponents
#' @export
ETScomponents= function(m){
    if (is.null(m$u))
        u = m$u
    else {
        if (is.vector(m$u)){
            u = matrix(m$u, 1, length(m$u))
        } else {
            nu = dim(m$u)
            u = as.numeric(m$u);
            u = matrix(u, nu[1], nu[2])
        }
    }
    output = ETSc("components", as.numeric(m$y), u, m$model, m$s, m$h,
                  m$criterion, m$armaIdent, m$identAll, m$forIntervals,
                  m$bootstrap, m$nSimul, m$verbose, m$lambda,
                  m$alphaL, m$betaL, m$gammaL, m$phiL, m$p0)
    if (is.ts(m$y))
        m$comp = ts(output$comp, start = start(m$y), frequency = frequency(m$y))
    else
        m$comp = output$comp
    colnames(m$comp) = strsplit(output$compNames, split = "/")[[1]]
    return(m)
}
    