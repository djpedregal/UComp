#' @title UCcommand
#' @description Auxiliar function for UC modeling
#'
#' @param command Command to execute: "forecast", "validate", "filter", "smooth", "disturb", "components", "all"
#' 
#' @param sys A \code{UComp} object created with \code{UC}
#' 
#' @return The input \code{UComp} object with the appropriate fields filled in
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{UC}}, \code{\link{UCforecast}}, \code{\link{UCvalidate}}, \code{\link{UCfilter}}, 
#'          \code{\link{UCsmooth}}, \code{\link{UCcomponents}}, \code{\link{UCdisturb}}
#'          
#' @examples
#' \donttest{
#' cycle <- UChp(USgdp)
#' plot(cycle)
#' }
#' @rdname UCcommand
#' @export
UCcommand = function(command, sys) {
    isnullu = FALSE
    if (is.null(sys$u)) {
        isnullu = TRUE
        sys$u = matrix(0, 1, 2)
    }
    if (sys$model == "error") {
        stop("ERROR: Model not specified or estimation failed!!\n")
        return(sys)
    }
    if (is.ts(sys$y)){
        seas = frequency(sys$y)
        y = as.numeric(sys$y)
    } else {
        y = sys$y
    }
    if (is.ts(sys$u)){
        u = as.numeric(sys$u)
    } else {
        u = sys$u
    }
    nu = dim(u)[2]
    kInitial = dim(u)[1]
    if (nu == 2){
        nu = length(sys$y) + sys$h
        kInitial = 0
    }
    if (any(sys$rhos < 0)) {
            if (any(sys$rhos == 1)) {
                    sys$periods = 1
                    sys$rhos = 1
            } else {
                    sys$periods = seas / (1 : floor(seas / 2))
                    sys$rhos = rep(1, length(sys$periods))
            }
    }
    output = UCompC(command, y, u, sys$model, sys$h, sys$lambda, sys$outlier, sys$tTest,
                    sys$criterion, sys$periods, sys$rhos, sys$verbose, sys$stepwise, sys$p0, sys$arma,
                    sys$TVP, sys$seas, sys$trendOptions, sys$seasonalOptions, sys$irregularOptions)
    if (output$model == "error"){
        sys$model = "error"
        return(sys);
    }
    if (is.ts(sys$y)){
        fY = frequency(sys$y)
        sY = start(sys$y, frequency = fY)
        aux = ts(matrix(NA, length(sys$y) + 1, 1), sY, frequency = fY)
        if (length(output$yFor > 0)){
            sys$yFor = ts(output$yFor, end(aux), frequency = fY)
            sys$yForV = ts(output$yForV, end(aux), frequency = fY)
        }
    } else {
        if (length(output$yFor > 0)){
            sys$yFor = output$yFor
            sys$yForV = output$yForV
        }
    }
    # Convert to R list
    # if (grepl("?", sys$model, fixed = TRUE)){
        sys$model = output$model
    # }
    sys$h = output$h
    sys$hidden$estimOk = output$estimOk
    sys$criteria = output$criteria;
    sys$lambda = output$lambda
    sys$periods = output$periods
    sys$rhos = output$rhos
    sys$p = as.vector(output$p)
    nPar = length(sys$p)
    names(sys$p) = output$parNames[1 : nPar]
    # sys$p0 = output$p0
    # Adapting periods of estimated cycles, rhos and model
    # ind = which(sys$rhos < 0)
    # if (length(ind) > 0) {
    #     # Changing cycle model
    #     pCycle = sys$periods[ind]
    #     mCycle = NULL
    #     for (i in 1 : length(pCycle)) {
    #         mCycle = paste0(mCycle, "+", round(pCycle[i], 2))
    #     }
    #     sys$model = sub("(/)[^/]+(/)", paste0("\\1", mCycle, "\\2"), sys$model)
    # }
 
    # if (any(sys$periods < 0)) {
    #     sys$model = MODEL
    #     sys$periods = PERIODS
    # }
    # print("line 65 UCestim")
    # print(sys$model)
    # print(MODEL)
    # print(sys$periods)
    if (is.ts(sys$y)){
        nNan = min(which(!is.na(y))) - 1
        initial = start(sys$y, frequency = frequency(y))
        if (nNan > 0){
            initial = end(ts(1 : (nNan + 1), start=initial, frequency=frequency(sys$y)))
        }
    }
    if (command == "validate" || command == "all") {
        sys$table = output$table
        sys$v = output$v
        sys$covp = output$covp
        sys$coef = as.vector(output$coef)
        nPar = length(sys$coef)
        rownames(sys$covp) = output$parNames[1 : dim(sys$covp)[1]]
        colnames(sys$covp) = output$parNames[1 : dim(sys$covp)[1]]
        names(sys$coef) = output$parNames[1 : nPar]
    }
    if (command == "components" || command == "all") {
        sys$comp = output$comp
        sys$compV = output$compV
        m = output$m  # + nCycles
        if (dim(u)[1] == 1 && dim(u)[2] == 2){
            k = 0
        } else {
            k = dim(u)[1]
        }
        nCycles = m - k - 4
        # Re-building matrices to their original sizes
        n = length(sys$comp) / m
        if (is.ts(sys$y)){
            sys$comp = ts(t(matrix(sys$comp, m, n)), start= initial, frequency = frequency(sys$y))
            sys$compV = ts(t(matrix(sys$compV, m, n)), start = initial, frequency = frequency(sys$y))
        } else {
            sys$comp = t(matrix(sys$comp, m, n))
            sys$compV = t(matrix(sys$compV, m, n))
        }
        if (length(size(sys$comp)) == 1){
            if (is.ts(sys$y)){
                sys$comp = ts(matrix(sys$comp, n + sys$h, 1), start = initial, frequency = frequency(sys$y))
                sys$compV = ts(matrix(sys$compV, n + sys$h, 1), start = initial, frequency = frequency(sys$y))
            } else {
                sys$comp = matrix(sys$comp, n + sys$h, 1)
                sys$compV = matrix(sys$compV, n + sys$h, 1)
            }
        }
        names1 = strsplit(output$compNames, "/")
        colnames(sys$comp) = names1[[1]][1 : ncol(sys$comp)]
        colnames(sys$compV) = names1[[1]][1 : ncol(sys$comp)]
    }
    if (command == "filter" || command == "smooth" || command == "disturb" || command == "all") {
        # n = length(sys$y) + sys$h
        # m = output$ns
        aux = dim(output$a)
        n = aux[2]
        m = aux[1]
        if (is.ts(sys$y)){
            fY = frequency(sys$y)
            sY = initial
            if (command == "disturb"){
                n = length(sys$y)
                mEta = length(output$eta) / n
                sys$eta = ts(t(matrix(output$eta, mEta, n)), sY, frequency = fY)
                sys$eps = ts(t(matrix(output$eps, 1, n)), sY, frequency = fY)
            } else {
                sys$a = ts(t(matrix(output$a, m, n)), sY, frequency = fY)
                # sys$aFor = ts(t(matrix(sys$aFor, m, sys$h)), end(aux), frequency = fY)
                sys$P = ts(t(matrix(output$P, m, n)), sY, frequency = fY)
                # sys$PFor = ts(t(matrix(sys$PFor, m, sys$h)), end(aux), frequency = fY)
                sys$yFit = ts(output$yFit, sY, frequency = fY)
                sys$yFitV = ts(output$yFitV, sY, frequency = fY)
                aux = ts(matrix(NA, length(sys$y) - length(output$v) + 1, 1), sY, frequency = fY)
                sys$v = ts(output$v, end(aux), frequency = fY)
            }
        } else {
            if (command == "disturb"){
                n = length(sys$y)
                sys$eta = t(matrix(output$eta, m, n))
                sys$eps = t(matrix(output$eps, 1, n))
            } else {
                sys$a = t(matrix(output$a, m, n))
                # sys$aFor = t(matrix(sys$aFor, m, sys$h))
                sys$P = t(matrix(output$P, m, n))
                # sys$PFor = t(matrix(sys$PFor, m, sys$h))
                sys$yFit = output$yFit
                sys$yFitV = output$yFitV
                sys$v = output$v
            }
        }
        if (command != "disturb"){
            names1 = strsplit(output$stateNames, "/")[[1]]
            nNames = length(names1)
            if (is.vector(sys$a)){
                sys$a = matrix(sys$a, length(sys$a), 1)
                sys$P = matrix(sys$P, length(sys$P), 1)
            }
            if (ncol(sys$a) >= nNames){
                sys$a = sys$a[, 1 : nNames]
                sys$P = sys$P[, 1 : nNames]
                if (length(size(sys$a)) == 1){
                    sys$a = matrix(sys$a, length(sys$a), 1)
                    sys$P = matrix(sys$P, length(sys$P), 1)
                }
                colnames(sys$a) = strsplit(output$stateNames, "/")[[1]]
                colnames(sys$P) = strsplit(output$stateNames, "/")[[1]]
            }
        }
    }
    if (isnullu)
         sys$u = NULL
    return(sys)
}
 

#' @title UCestim
#' @description Estimates and forecasts UC models
#'
#' @details \code{UCestim} estimates and forecasts a time series using an
#' UC model.
#' The optimization method is a BFGS quasi-Newton algorithm with a 
#' backtracking line search using Armijo conditions.
#' Parameter names in output table are the following:
#' \itemize{
#' \item Damping:   Damping factor for DT trend.
#' \item Level:     Variance of level disturbance.
#' \item Slope:     Variance of slope disturbance.
#' \item Rho(#):    Damping factor of cycle #.
#' \item Period(#): Estimated period of cycle #.
#' \item Var(#):    Variance of cycle #.
#' \item Seas(#):   Seasonal harmonic with period #.
#' \item Irregular: Variance of irregular component.
#' \item AR(#):     AR parameter of lag #.
#' \item MA(#):     MA parameter of lag #.
#' \item AO#:       Additive outlier in observation #.
#' \item LS#:       Level shift outlier in observation #.
#' \item SC#:       Slope change outlier in observation #.
#' \item Beta(#):   Beta parameter of input #.
#' \item Cnst:      Constant.
#' }
#' 
#' Standard methods applicable to UComp objects are print, summary, plot,
#' fitted, residuals, logLik, AIC, BIC, coef, predict, tsdiag.
#'
#' @param sys an object of type \code{UComp} created with \code{UC}
#' 
#' @return The same input object with the appropriate fields 
#' filled in, in particular:
#' \itemize{
#' \item p:        Estimated transformed parameters
#' \item v:        Estimated innovations (white noise in correctly specified models)
#' \item yFor:     Forecast values of output
#' \item yForV:    Forecasted values variance
#' \item criteria: Value of criteria for estimated model
#' \item covp:     Covariance matrix of estimated transformed parameters
#' \item grad:     Gradient of log-likelihood at the optimum
#' \item iter:     Estimation iterations
#' }
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{UC}}, \code{\link{UCforecast}}, \code{\link{UCvalidate}}, \code{\link{UCfilter}}, 
#'          \code{\link{UCsmooth}}, \code{\link{UCdisturb}}, \code{\link{UCcomponents}},
#'          \code{\link{UChp}}
#'          
#' @examples
#' \donttest{
#' m1 <- UCsetup(log(AirPassengers))
#' m1 <- UCestim(m1)
#' }
#' @rdname UCestim
#' @export
UCestim = function(sys){
    sys = UCcommand("forecast", sys)
    return(sys)
}

#' @title UCvalidate
#' @description Shows a table of estimation and diagnostics results for UC models.
#' Equivalent to print or summary.
#' The table shows information in four sections:
#' Firstly, information about the model estimated, the relevant 
#' periods of the seasonal component included, and further information about
#' convergence.
#' Secondly, parameters with their names are provided, the asymptotic standard errors, 
#' the ratio of the two, and the gradient at the optimum. One asterisk indicates 
#' concentrated-out parameters and two asterisks signals parameters constrained during estimation.
#' Thirdly, information criteria and the value of the log-likelihood.
#' Finally, diagnostic statistics about innovations, namely, the Ljung-Box Q test of absense
#' of autocorrelation statistic for several lags, the Jarque-Bera gaussianity test, and a
#' standard ratio of variances test.
#'
#' @param sys an object of type \code{UComp} created with \code{UC}
#' @param printScreen print to screen or just return output table
#' 
#' @return The same input object with the appropriate fields 
#' filled in, in particular:
#' \itemize{
#' \item table: Estimation and validation table
#' }
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{UC}}, \code{\link{UCforecast}}, \code{\link{UCfilter}}, 
#'          \code{\link{UCsmooth}}, \code{\link{UCdisturb}}, \code{\link{UCcomponents}},
#'          \code{\link{UChp}}
#'          
#' @examples
#' \donttest{
#' m1 <- UC(log(gdp))
#' m1 <- UCvalidate(m1)
#' }
#' @rdname UCvalidate
#' @export
UCvalidate = function(sys, printScreen = TRUE){
    VERBOSE = sys$verbose
    sys$verbose = FALSE
    sys = UCcommand("validate", sys)
    sys$verbose = VERBOSE
    if (printScreen){
        cat(sys$table)
    }
    return(sys)
}

#' @title UChp
#' @description Hodrick-Prescott filter estimation
#'
#' @param y A time series object
#' 
#' @param lambda Smoothing constant (default: 1600)
#' 
#' @return The cycle estimation
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{UC}}, \code{\link{UCforecast}}, \code{\link{UCvalidate}}, \code{\link{UCfilter}}, 
#'          \code{\link{UCsmooth}}, \code{\link{UCcomponents}}, \code{\link{UCdisturb}}
#'          
#' @examples
#' \donttest{
#' cycle <- UChp(USgdp)
#' plot(cycle)
#' }
#' @rdname UChp
#' @export
UChp = function(y, lambda = 1600){
    m = UCsetup(y, model = "irw/none/arma(0,0)")
    m$hidden$truePar = c(log(1 / lambda) / 2, 0)
    m = UCcomponents(m)
    return(y - m$comp[, 1])
}

#' @title UCcomponents
#' @description Estimates unobserved components of UC models
#' Standard methods applicable to UComp objects are print, summary, plot,
#' fitted, residuals, logLik, AIC, BIC, coef, predict, tsdiag.
#'
#' @param sys an object of type \code{UComp} created with \code{UC} or \code{UCforecast}
#' 
#' @return The same input object with the appropriate fields 
#' filled in, in particular:
#' \itemize{
#' \item comp:  Estimated components in matrix form
#' \item compV: Estimated components variance in matrix form
#' }
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{UC}}, \code{\link{UCforecast}}, \code{\link{UCvalidate}}, \code{\link{UCfilter}}, 
#'          \code{\link{UCsmooth}}, \code{\link{UCdisturb}},
#'          \code{\link{UChp}}
#'          
#' @examples
#' \donttest{
#' m1 <- UC(log(AirPassengers))
#' m1 <- UCcomponents(m1)
#' }
#' @rdname UCcomponents
#' @export
UCcomponents= function(sys){
    VERBOSE = sys$verbose
    sys$verbose = FALSE
    sys = UCcommand("components", sys)
    sys$verbose = VERBOSE
    return(sys)
}

#' @title UCfilter
#' @description Runs the Kalman Filter for UC models
#' Standard methods applicable to \code{UComp} objects are print, summary, plot,
#' fitted, residuals, logLik, AIC, BIC, coef, predict, tsdiag.
#'
#' @param sys an object of type \code{UComp} created with \code{UC}
#' 
#' @return The same input object with the appropriate fields 
#' filled in, in particular:
#' \itemize{
#' \item yFit:  Fitted values of output
#' \item yFitV: Variance of fitted values of output
#' \item a:     State estimates
#' \item P:     Variance of state estimates (diagonal of covariance matrices)
#' }
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{UC}}, \code{\link{UCforecast}}, \code{\link{UCvalidate}}, 
#'          \code{\link{UCsmooth}}, \code{\link{UCdisturb}}, \code{\link{UCcomponents}},
#'          \code{\link{UChp}}
#'          
#' @examples
#' \donttest{
#' m1 <- UC(log(AirPassengers))
#' m1 <- UCfilter(m1)
#' }
#' @rdname UCfilter
#' @export
UCfilter = function(sys){
    VERBOSE = sys$verbose
    sys$verbose = FALSE
    sys = UCcommand("filter", sys)
    sys$verbose = VERBOSE
    return(sys)
}

#' @title UCsmooth
#' @description Runs the Fixed Interval Smoother for UC models
#' Standard methods applicable to \code{UComp} objects are print, summary, plot,
#' fitted, residuals, logLik, AIC, BIC, coef, predict, tsdiag.
#'
#' @param sys an object of type \code{UComp} created with \code{UC}
#' 
#' @return The same input object with the appropriate fields 
#' filled in, in particular:
#' \itemize{
#' \item yFit:  Fitted values of output
#' \item yFitV: Variance of fitted values of output
#' \item a:     State estimates
#' \item P:     Variance of state estimates (diagonal of covariance matrices)
#' }
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{UC}}, \code{\link{UCforecast}}, \code{\link{UCvalidate}}, \code{\link{UCfilter}}, 
#'          \code{\link{UCdisturb}}, \code{\link{UCcomponents}},
#'          \code{\link{UChp}}
#'          
#' @examples
#' \donttest{
#' m1 <- UC(log(AirPassengers))
#' m1 <- UCsmooth(m1)
#' }
#' @rdname UCsmooth
#' @export
UCsmooth = function(sys){
    VERBOSE = sys$verbose
    sys$verbose = FALSE
    sys = UCcommand("smooth", sys)
    sys$verbose = VERBOSE
    return(sys)
}

#' @title UCdisturb
#' @description Runs the Disturbance Smoother for UC models
#' Standard methods applicable to \code{UComp} objects are print, summary, plot,
#' fitted, residuals, logLik, AIC, BIC, coef, predict, tsdiag.
#'
#' @param sys an object of type \code{UComp} created with \code{UC}
#' 
#' @return The same input object with the appropriate fields 
#' filled in, in particular:
#' \itemize{
#' \item yFit:  Fitted values of output
#' \item yFitV: Variance of fitted values of output
#' \item a:     State estimates
#' \item P:     Variance of state estimates (diagonal of covariance matrices)
#' \item eta:   State perturbations estimates
#' \item eps:   Observed perturbations estimates
#' }
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{UC}}, \code{\link{UCforecast}}, \code{\link{UCvalidate}}, \code{\link{UCfilter}}, 
#'          \code{\link{UCsmooth}}, \code{\link{UCcomponents}},
#'          \code{\link{UChp}}
#'          
#' @examples
#' \donttest{
#' m1 <- UC(log(AirPassengers))
#' m1 <- UCdisturb(m1)
#' }
#' @rdname UCdisturb
#' @export
UCdisturb = function(sys){
    VERBOSE = sys$verbose
    sys$verbose = FALSE
    sys = UCcommand("disturb", sys)
    sys$verbose = VERBOSE
    return(sys)
}
