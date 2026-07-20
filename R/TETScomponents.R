#' @title TETScomponents
#' @description Estimates components of TOBIT TETS models
#'
#' @param m an object of type \code{TETS} created with \code{TETSforecast}
#' 
#' @return The same input object with the appropriate fields 
#' filled in, in particular:
#' \item{comp}{Estimated components in matrix form}
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{TETS}}, \code{\link{TETSforecast}}, \code{\link{TETSvalidate}}
#'        
#' @examples
#' \donttest{
#' m1 <- TETS(log(gdp))
#' m1 <- TETScomponents(m1)
#' }
#' @rdname TETScomponents
#' @export
TETScomponents= function(m){
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
    if (inherits(m, "ETS")){
        output = ETSc("components", as.numeric(m$y), u, m$model, m$s, m$h,
                  m$criterion, m$armaIdent, m$identAll, m$forIntervals,
                  m$bootstrap, m$nSimul, m$verbose, m$lambda,
                  m$alphaL, m$betaL, m$gammaL, m$phiL, m$p0)
    } else {
        output = TETSc("components", as.numeric(m$y), u, m$model, m$s, m$h,
                  m$criterion, m$armaIdent, m$identAll, m$forIntervals,
                  m$bootstrap, m$nSimul, m$verbose, m$lambda,
                  m$alphaL, m$betaL, m$gammaL, m$phiL, m$p0, m$Ymin, m$Ymax)
    }
    if (is.ts(m$y))
        m$comp = ts(output$comp, start = start(m$y), frequency = frequency(m$y))
    else
        m$comp = output$comp
    colnames(m$comp) = strsplit(output$compNames, split = "/")[[1]]
    return(m)
}
    