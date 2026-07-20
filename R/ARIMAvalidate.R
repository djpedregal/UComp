#' @title ARIMAvalidate
#' @description Shows a table of estimation and diagnostics results for ARIMA models
#'
#' @param m an object of type \code{ARIMA} created with \code{ARIMAforecast}
#' 
#' @return The same input object with the appropriate fields 
#' filled in, in particular:
#' \item{table}{Estimation and validation table}
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{ARIMA}}, \code{\link{ARIMAforecast}}, \code{\link{ARIMAvalidate}},
#'          
#' @examples
#' \donttest{
#' m1 <- ARIMAforecast(log(gdp))
#' m1 <- ARIMAvalidate(m1)
#' }
#' @rdname ARIMAvalidate
#' @export
ARIMAvalidate = function(m){
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
    if (sum(m$model) == 0)
            m$model = NULL
    output = ARIMAc("validate", as.numeric(m$y), u, m$model, m$cnst, m$s, m$criterion, 
                    m$h, m$verbose, m$lambda, m$maxOrders, m$bootstrap, m$nSimul,
                    m$fast, m$identDiff, m$identMethod)
    if (length(output) == 1){   # ERROR!!
            stop()
    }
    m$table = output$table
    m$error = output$error
    m$v = output$error
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
    # Buscando test heterocedasticidad con valor p NaN
    # ind = which(grepl("nan", m$table))
    # if (any(ind)){
    #     for (i in 1 : length(ind)){
    #         line = m$table[ind[i]]
    #         df = as.numeric(substr(line, 9, 12))
    #         Fstat = as.numeric(substr(line, 15, 31))
    #         pval = round(pf(Fstat, df, df), 4)
    #         line = gsub("   nan", pval, line)
    #         m$table[ind[i]] = line
    #     }
    # }
    # if (m$verbose)
    #     cat(m$table)
    return(m)
}
    