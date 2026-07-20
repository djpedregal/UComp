#' @title TETSestim
#' @description Estimates and forecasts TOBIT TETS models
#'
#' @details \code{TETSestim} estimates and forecasts a time series using an
#' a TOBIT TETS model
#'
#' @param m an object of type \code{TETS} created with \code{TETSforecast}
#' 
#' @return The same input object with the appropriate fields 
#' filled in, in particular:
#' \item{p}{Estimated parameters}
#' \item{yFor}{Forecasted values of output}
#' \item{yForV}{Variance of forecasted values of output}
#' \item{ySimul}{Bootstrap simulations for forecasting distribution evaluation}
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{TETS}}, \code{\link{TETSforecast}}, \code{\link{TETSvalidate}},
#'          \code{\link{TETScomponents}}
#'          
#' @examples
#' \donttest{
#' m1 <- TETSsetup(log(gdp))
#' m1 <- TETSestim(m1)
#' }
#' @rdname TETSestim
#' @export
TETSestim = function(m){
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
    output = TETSc("estimate", as.numeric(m$y), u, m$model, m$s, m$h,
                  m$criterion, m$armaIdent, m$identAll, m$forIntervals,
                  m$bootstrap, m$nSimul, m$verbose, m$lambda,
                  m$alphaL, m$betaL, m$gammaL, m$phiL, m$p0, m$Ymin, m$Ymax)
    if (length(output) == 1){   # ERROR!!
            stop()
    } else {
            m$model = output$model
            m$lambda = output$lambda
            m$p = output$p
            m$truep = output$truep
            lu = size(m$u)[1]
            if (lu > 0)
                m$h = lu - length(m$y)
            if (is.ts(m$y) && m$h > 0){
                fake = ts(c(m$y, NA), start = start(m$y), frequency = frequency(m$y))
                m$yFor = ts(output$yFor, start = end(fake), frequency = frequency(m$y))
                m$yForV = ts(output$yForV, start = end(fake), frequency = frequency(m$y))
                if (m$bootstrap)
                    m$ySimul = ts(output$ySimul, start = end(fake), frequency = frequency(m$y))
            } else if (m$h > 0) {
                m$yFor = output$yFor
                m$yForV = output$yForV
                m$ySimul = output$ySimul
            }
            m$criteria = output$criteria
            return(m)
    }
}