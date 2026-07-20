#' @title ARIMAestim
#' @description Estimates and forecasts ARIMA models
#'
#' @details \code{ARIMAestim} estimates and forecasts a time series using 
#' an ARIMA model
#'
#' @param m an object of type \code{ARIMA} created with \code{ARIMAforecast}
#' 
#' @return The same input object with the appropriate fields 
#' filled in, in particular:
#' \item{p}{Estimated parameters}
#' \item{yFor}{Forecasted values of output}
#' \item{yForV}{Variance of forecasted values of output}
#' \item{ySimul}{Bootstrap simulations for forecasting distribution evaluation}
#' 
#' @export
#' 
ARIMAestim = function(m){
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
    output = ARIMAc("estimate", as.numeric(m$y), u, m$model, m$cnst, m$s, m$criterion,
                    m$h, m$verbose, m$lambda, m$maxOrders, m$bootstrap, m$nSimul,
                    m$fast, m$identDiff, m$identMethod)
    if (length(output) == 1){   # ERROR!!
            stop()
    } else {
            m$p = output$p
            m$ySimul = output$ySimul
            m$lambda = output$lambda
            m$model = output$orders
            m$cnst = output$cnst
            m$u = output$u
            m$BIC = output$BIC
            m$AIC = output$AIC
            m$AICc = output$AICc
            m$IC = output$IC
            # m$betaAug = output$betaAug
            # m$betaAugVar = output$betaAugVar
            if (!is.null(dim(m$u))){
                    lu = dim(m$u)[1]
                    if (lu == 1)
                            lu = dim(m$u)[2]
            } else {
                    lu = length(m$u)
            }
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
            return(m)
    }
}