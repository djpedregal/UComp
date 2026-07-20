#' @title TETSvalidate
#' @description Shows a table of estimation and diagnostics results for TOBIT TETS models
#'
#' @param m an object of type \code{TETS} created with \code{TETSforecast}
#' 
#' @return The same input object with the appropriate fields 
#' filled in, in particular:
#' \item{table}{Estimation and validation table}
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{TETS}}, \code{\link{TETSforecast}}, \code{\link{TETSvalidate}},
#'          \code{\link{TETScomponents}}
#'          
#' @examples
#' \donttest{
#' m1 <- TETSforecast(log(gdp))
#' m1 <- TETSvalidate(m1)
#' }
#' @rdname TETSvalidate
#' @export
TETSvalidate = function(m){
    # suppressWarnings()
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
            output = ETSc("validate", as.numeric(m$y), u, m$model, m$s, m$h,
                          m$criterion, m$armaIdent, m$identAll, m$forIntervals,
                          m$bootstrap, m$nSimul, m$verbose, m$lambda,
                          m$alphaL, m$betaL, m$gammaL, m$phiL, m$p0)
    } else {
            output = TETSc("validate", as.numeric(m$y), u, m$model, m$s, m$h,
                          m$criterion, m$armaIdent, m$identAll, m$forIntervals,
                          m$bootstrap, m$nSimul, m$verbose, m$lambda,
                          m$alphaL, m$betaL, m$gammaL, m$phiL, m$p0, m$Ymin, m$Ymax)
    }
    if (length(output) == 1){   # ERROR!!
        stop()
    } else {
        if (is.ts(m$y))
            m$comp = ts(output$comp, start = start(m$y), frequency = frequency(m$y))
        else
            m$comp = output$comp
        m$table = output$table
        # Buscando test heterocedasticidad con valor p NaN
        # ind = which(grepl("nan", m$table))
        # if (any(ind)){
        #     for (i in 1 : length(ind)){
        #         line = m$table[ind[i]]
        #         Fstat = as.numeric(substr(line, 15, 31))
        #         if (grepl("Bera", line)) {
        #             pval = round(pchisq(Fstat, 2), 4)
        #         } else if (grepl("H(", line)) {
        #             df = as.numeric(substr(line, 9, 12))
        #             pval = round(pf(Fstat, df, df), 4)
        #         }
        #         line = gsub("   nan", pval, line)
        #         m$table[ind[i]] = line
        #     }
        # }
        # if (m$verbose)
        #     cat(m$table)
        colnames(m$comp) = strsplit(output$compNames, split = "/")[[1]]
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
    