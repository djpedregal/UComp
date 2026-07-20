#' Daily calendar harmonic phases (internal utility)
#'
#' Internal function used to construct monthly and annual harmonic
#' seasonal phases based on a daily calendar, accounting for leap years.
#'
#' This function is used internally by \code{DAYforecast} and
#' \code{WEEKforecast} to generate deterministic seasonal regressors.
#'
#' @noRd
daily_time = function(n, initial_date) {
        # ciclo mensual
        t28 = 2 * pi * (1 : 28) / 28
        t29 = 2 * pi * (1 : 29) / 29
        t30 = 2 * pi * (1 : 30) / 30
        t31 = 2 * pi * (1 : 31) / 31
        year = c(t31, t28, t31, t30, t31, t30, t31, t31, t30, t31, t30, t31)
        yearb = c(t31, t29, t31, t30, t31, t30, t31, t31, t30, t31, t30, t31)
        # Secuencia de fechas
        dates <- seq.Date(as.Date(initial_date), by = "day", length.out = n)
        # Año y día del año
        yr  <- as.integer(format(dates, "%Y"))
        doy <- as.integer(format(dates, "%j"))
        # Años bisiestos
        is_leap <- (yr %% 4 == 0 & yr %% 100 != 0) | (yr %% 400 == 0)
        # Vector final
        month = ifelse(is_leap, yearb[doy], year[doy])
        t365 = 2 * pi * (1 : 365) / 365
        t366 = 2 * pi * (1 : 366) / 366
        year = ifelse(is_leap, t366, t365)
        return(list(month = month, year = year))
}
#' @title DAYforecast
#' @description Estimates and forecasts daily time series with weekly and monthly seasonal patterns
#'
#' @details \code{DAYforecast} estimates and forecasts daily univariate time series using
#' Dynamic Harmonic Regression (DHR).
#' 
#' The function is tailored for daily data and explicitly accounts for:
#' \itemize{
#' \item Monthly seasonality through sinusoidal terms based on calendar months
#' \item Weekly seasonality through sinusoidal terms with periods of 7 days
#' }
#'
#' Standard methods applicable to \code{UComp} objects are available.
#'
#' @param y A numeric vector, \code{ts} object or list containing daily observations of the
#'          dependent variable.
#' @param u Optional matrix or vector of exogenous regressors. If provided, it must cover
#'          both the estimation and forecast periods.
#' @param initial_date A character or \code{Date} object indicating the calendar start date
#'          of the series.
#' @param h Integer indicating the forecast horizon (number of days ahead).
#' @param lambda Box–Cox transformation parameter (NULL for automatic identification).
#' @param criterion Information criterion used for model selection (AIC, BIC, AICc).
#' @param p0 Initial parameter vector for Maximum Likelihood estimation.
#' @param verbose if \code{TRUE}, prints estimation details.
#'
#' @return An object of class \code{UComp}. It is a list containing the model specification,
#'         estimated parameters, and forecasting results.
#'         
#' @author Diego J. Pedregal
#'
#' @seealso \code{\link{UC}}, \code{\link{UCforecast}}, \code{\link{UCfilter}},
#'          \code{\link{UCsmooth}}, \code{\link{UCdisturb}},
#'          \code{\link{UCcomponents}}
#'
#' @examples
#' \donttest{
#' y <- rnorm(400)
#' m <- DAYforecast(y, initial_date = "2015-01-01", h = 30)
#' }
#'
#' @rdname DAYforecast
#' @export
DAYforecast = function(y, u = NULL, initial_date, h = 24, lambda = 1, criterion = "aic",
                       p0 = -9999.9, verbose = FALSE){
        # Standard daily model with outliers and frequency=7
        if (is.ts(y)) {
                y = ts(y, start(y), frequency = 7)
        } else if (is.list(y)) {
                y = ts(y[, "y"], frequency = 7)
        } else {
                y = ts(y, frequency = 7)
        }
        mo = UCforecast(y, outlier=4, lambda=lambda, h=h)
        # Daily data with weekly and monthly cycles (DHR)
        if (is.ts(y)) {
                y = ts(y, frequency = 1)
        } else if (is.list(y)) {
                y = ts(y[, "y"], frequency = 1)
        }
        n = length(y)
        if (is.null(u)) {
                nu = 0
        } else if (is.vector(u)) {
                u = matrix(u, length(u), 1)
                h = nrow(u) - n
                nu = 1
        } else if (ncol(u) > nrow(u)) {
                u = t(u)
                h = nrow(u) - n
                nu = ncol(u)
        }
        n <- length(y) + h
        ty = daily_time(n, initial_date)$month
        t = (1 : n) * 2 * pi / 7
        u1 = cbind(u, sin(ty), cos(ty), 
                   sin(ty * 2), cos(ty * 2), 
                   sin(ty * 3), cos(ty * 3),
                   sin(t), cos(t), 
                   sin(2 * t), cos(2 * t), 
                   sin(3 * t), cos(3 * t))
        TVP = c(rep(0, nu), rep(1, 6), rep(2, 6))
        m = UCforecast(y, 
                       u1, 
                       model= "?/?/arma(0,0)",
                       lambda=lambda, 
                       criterion=criterion, 
                       p0=p0, 
                       TVP=TVP, 
                       verbose=verbose)
        if (is.null(u)) {
                m$u = u
                mo$u = u
        }
        if (mo$criteria[2] < m$criteria[2]) {
                return(mo)
        } else {
                return(m)
        }
}
#' @title WEEKforecast
#' @description Estimates and forecasts weekly time series with monthly and annual seasonal patterns
#'
#' @details \code{WEEKforecast} estimates and forecasts univariate weekly time series using
#' Dynamic Harmonic Regression (DHR).
#'
#' The function is designed for weekly data but builds the seasonal structure from
#' underlying daily calendar information. In particular, it incorporates:
#' \itemize{
#' \item Monthly seasonality via harmonic terms derived from calendar months
#' \item Annual seasonality via multiple harmonic components based on calendar years
#' }
#'
#' Standard methods applicable to \code{UComp} objects are available.
#'
#' @param y A numeric vector, \code{ts} object or list containing weekly observations of the
#'          dependent variable.
#' @param u Optional matrix or vector of exogenous regressors. If provided, it must cover
#'          both the estimation and forecast periods.
#' @param initial_date A character or \code{Date} object indicating the calendar start date
#'          of the series.
#' @param h Integer indicating the forecast horizon (number of weeks ahead).
#' @param lambda Box–Cox transformation parameter.
#' @param criterion Information criterion used for model selection (AIC, BIC or AICc).
#' @param p0 Initial parameter vector for Maximum Likelihood estimation.
#' @param verbose if \code{TRUE}, prints estimation details.
#'
#' @return An object of class \code{UComp}. It is a list containing the model specification,
#'         estimated parameters, and forecasting results.
#'
#' @author Diego J. Pedregal
#'
#' @seealso \code{\link{UC}}, \code{\link{UCforecast}}, \code{\link{UCfilter}},
#'          \code{\link{UCsmooth}}, \code{\link{UCdisturb}},
#'          \code{\link{UCcomponents}}
#'
#' @examples
#' \donttest{
#' y <- rnorm(400)
#' m <- WEEKforecast(y, initial_date = "2015-01-01", h = 12)
#' }
#'
#' @rdname WEEKforecast
#' @export
WEEKforecast = function(y, u = NULL, initial_date, h = 24, lambda = 1, criterion = "aic",
                        p0 = -9999.9, verbose = FALSE){
        # Standard daily model and frequency=7
        if (is.ts(y)) {
                y = ts(y, start(y), frequency = 4.28)
        } else if (is.list(y)) {
                y = ts(y[, "y"], frequency = 4.28)
        } else {
                y = ts(y, frequency = 4.28)
        }
        mo = UCforecast(y, model="?/?/arma(0,0)", lambda=lambda, h=h)
        # Standard daily model with outliers and frequency=7
        if (is.ts(y)) {
                y = ts(y, start(y), frequency = 52)
        } else if (is.list(y)) {
                y = ts(y[, "y"], frequency = 52)
        } else {
                y = ts(y, frequency = 52)
        }
        ma = UCforecast(y, model="?/?/arma(0,0)", lambda=lambda, h=h)
        # Weekly forecast with monthly and annual cycles (DHR)
        if (is.ts(y)) {
                y = ts(y, frequency = 1)
        } else if (is.list(y)) {
                y = ts(y[, "y"], frequency = 1)
        }
        n = length(y) * 7
        if (is.null(u)) {
                nu = 0
        } else if (is.vector(u)) {
                u = matrix(u, length(u), 1)
                h = nrow(u) - n
                nu = 1
        } else if (ncol(u) > nrow(u)) {
                u = t(u)
                h = nrow(u) - n
                nu = ncol(u)
        }
        n <- (length(y) + h) * 7
        aux = daily_time(n, initial_date)
        # Muestreo semanal
        ind = seq(1, n, 7)
        tm = aux$month[ind]
        ty = aux$year[ind]
        u = cbind(u, sin(tm), cos(tm),
                  sin(2 * tm), cos(2 * tm),
                  sin(ty), cos(ty),
                  sin(ty * 2), cos(ty * 2),
                  sin(ty * 3), cos(ty * 3),
                  sin(ty * 4), cos(ty * 4),
                  sin(ty * 5), cos(ty * 5),
                  sin(ty * 6), cos(ty * 6))
        TVP = c(rep(0, nu), rep(1, 4), rep(2, 12))
        TVP = c(rep(0, nu), 1 : 16)
        m = UCforecast(y, 
                       u, 
                       model= "?/none/arma(0,0)",
                       lambda=lambda, 
                       criterion=criterion, 
                       p0=p0, 
                       TVP=TVP, 
                       periods=1, 
                       verbose=verbose)
        if (is.null(u)) {
                m$u = u
                mo$u = u
        }
        model = mo
        if (ma$criteria[2] < mo$criteria[2])
                model = ma
        if (m$criteria[2] < ma$criteria[2])
                model = m
        return(model)
}

