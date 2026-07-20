#' @title size
#' @description Size of vector, matrix or array
#'
#' @param y a vector, matrix or array
#'
#' @return A vector with all the dimensions
#'
#' @author Diego J. Pedregal
#'
#' @seealso \code{\link{colMedians}}, \code{\link{rowMedians}}, \code{\link{tests}},
#'          \code{\link{sumStats}}, \code{\link{gaussTest}}, \code{\link{ident}},
#'          \code{\link{cusum}}, \code{\link{varTest}}, \code{\link{conv}},
#'          \code{\link{armaFilter}}, \code{\link{dif}}, \code{\link{roots}},
#'          \code{\link{zplane}}, \code{\link{acft}}, \code{\link{slide}},
#'          \code{\link{plotSlide}}, \code{\link{Accuracy}}, \code{\link{tsDisplay}}
#'
#' @examples
#' s <- size(matrix(4, 3, 2))
#' s <- size(rep(4, 3))
#' s <- size(array(4, c(3, 2, 2)))
#' @rdname size
#' @export
size = function(y){
  # Size of any vector or matrix
  out = dim(y)
  if (is.null(out))
    out = length(y)
  return(out)
}
#' @title plus_one
#' @description Returns date of next to end time series y
#'
#' @param y a ts object
#'
#' @return Next time stamp
#'
#' @author Diego J. Pedregal
#'
#' @rdname plus_one
#' @export
plus_one = function(y){
  # Returns date of next to end time series y
  # y is a ts object
  return(end(ts(c(1, 2), start=end(y), frequency=frequency(y))))
}
#' @title extract
#' @description Reorder data frame returning column col reordered
#' according to the values in column accordingTo
#'
#' @param x a data frame
#' @param col column to be ordered
#' @param accordingTo column to take as the pattern
#'
#' @return Data frame reordered accoring to a given column data
#'
#' @author Diego J. Pedregal
#'
#' @rdname extract
#' @export
extract = function(x, col, accordingTo = 1){
  # Reorder data frame returning column col reordered according to
  # the values in column accordingTo
  y = as.data.frame(x)
  values = unique(y[, accordingTo])
  n = nrow(y)
  out = matrix(y[, col], n / length(values), length(values))
  colnames(out) = values
  return(out)
}
#' @title colMedians
#' @description Medians of matrix by columns
#'
#' @param x a matrix
#' @param na.rm boolean indicating whether to remove nans
#' @param ... rest of inputs
#'
#' @return A vector with all the medians in columns
#'
#' @author Diego J. Pedregal
#'
#' @seealso \code{\link{rowMedians}}, \code{\link{tests}},
#'          \code{\link{sumStats}}, \code{\link{gaussTest}}, \code{\link{ident}},
#'          \code{\link{cusum}}, \code{\link{varTest}}, \code{\link{conv}},
#'          \code{\link{armaFilter}}, \code{\link{dif}}, \code{\link{roots}},
#'          \code{\link{zplane}}, \code{\link{acft}}, \code{\link{slide}},
#'          \code{\link{plotSlide}}, \code{\link{Accuracy}}, \code{\link{tsDisplay}},
#'          \code{\link{size}}
#'
#' @examples
#' s <- colMedians(matrix(4, 3, 2))
#' @rdname colMedians
#' @export
colMedians = function(x, na.rm = TRUE, ...){
  # Median of matrix columns
  colx = dim(x)[2]
  out = rep(NA, colx)
  for (i in 1 : colx){
    out[i] = median(x[, i], na.rm = na.rm, ...)
  }
  return(out)
}
#' @title rowMedians
#' @description Medians of matrix by rows
#'
#' @param x a matrix
#' @param na.rm boolean indicating whether to remove nans
#' @param ... rest of inputs
#'
#' @return A vector with all the medians in rows
#'
#' @author Diego J. Pedregal
#'
#' @seealso \code{\link{colMedians}}, \code{\link{tests}},
#'          \code{\link{sumStats}}, \code{\link{gaussTest}}, \code{\link{ident}},
#'          \code{\link{cusum}}, \code{\link{varTest}}, \code{\link{conv}},
#'          \code{\link{armaFilter}}, \code{\link{dif}}, \code{\link{roots}},
#'          \code{\link{zplane}}, \code{\link{acft}}, \code{\link{slide}},
#'          \code{\link{plotSlide}}, \code{\link{Accuracy}}, \code{\link{tsDisplay}},
#'          \code{\link{size}}
#'
#' @examples
#' s <- rowMedians(matrix(4, 3, 2))
#' @rdname rowMedians
#' @export
rowMedians = function(x, na.rm = TRUE, ...){
  # Median of matrix rows
  return(colMedians(t(x), na.rm = na.rm, ...))
}
#' @title removeNaNs
#' @description Remove nans at beginning or end of vector
#'
#' @param x a vector or a ts object
#'
#' @return vector with nans removed (only those at beginning or end)
#'
#' @author Diego J. Pedregal
#'
#' @rdname removeNaNs
#' @export
removeNaNs = function(x){
    x = as.ts(x)
    t = time(x)
    ind = which(!is.na(x))
    ind1 = min(ind)
    ind2 = max(ind)
    return(ts(x[ind1 : ind2], start = t[ind1], frequency = frequency(x)))
}
#' @title tests
#' @description Tests on a time series
#'
#' @details Multiple tests on a time series, including summary statistics,
#' autocorrelation, Gaussianity and heteroskedasticity,
#'
#' @param y a vector, ts or tsibble object
#' @param parts proportion of sample to include in ratio of variances test
#' @param nCoef number of autocorrelation coefficients to estimate
#' @param nPar number of parameters in a model if y is a residual
#' @param s seasonal period, number of observations per year
#' @param avoid number of observations to avoid at beginning of sample to
#' eliminate initial effects
#'
#' @return Table with all test results
#' 
#' @author Diego J. Pedregal
#'
#' @seealso \code{\link{colMedians}}, \code{\link{rowMedians}},
#'          \code{\link{sumStats}}, \code{\link{gaussTest}}, \code{\link{ident}},
#'          \code{\link{cusum}}, \code{\link{varTest}}, \code{\link{conv}},
#'          \code{\link{armaFilter}}, \code{\link{dif}}, \code{\link{roots}},
#'          \code{\link{zplane}}, \code{\link{acft}}, \code{\link{slide}},
#'          \code{\link{plotSlide}}, \code{\link{Accuracy}}, \code{\link{tsDisplay}},
#'          \code{\link{size}}
#'
#' @examples
#' tests(AirPassengers)
#' @rdname tests
#' @export
tests = function(y,
                 parts = 1 / 3,
                 nCoef = min(25, length(x) / 4),
                 nPar = 0,
                 s = frequency(y), avoid = 16){
  # Statistical tests for time series
  if (is.list(y) && !is_tsibble(y)){
      x = tail(as.ts(residuals(y)), -avoid)
      s = frequency(x)
  } else {
      x = as.ts(y)
  }
  #x = removeNaNs(x)
  sumS = capture.output(sumStats(x))
  nCoef = max(floor(nCoef), 1)
  LB = ident(x, nCoef, nPar, TRUE)
  LBs = capture.output(LB)
  pLB = plotAcfPacf(LB$SACF, LB$SPACF, s, length(x), TRUE)
  pGAUSS = gaussTest(x, TRUE)
  pGs = capture.output(pGAUSS[[3]])
  pCUSUM = cusum(x, TRUE)
  pVar = varTest(x, parts)
  pVars = capture.output(pVar)
  out = c("Summary statistics:\n",
              "==================\n",
              sumS,
              "\n",
              "Autocorrelation tests:\n",
              "=====================\n",
              LBs,
              "\n",
              "Gaussianity tests:\n",
              "=================\n",
              pGs,
              "\n",
              "Ratio of variance tests:\n",
              "=======================\n",
              pVars)
  for (i in 1 : length(out)){
      if (substring(out[i], nchar(out[i])) != "\n"){
          out[i] = paste0(out[i], "\n")
      }
  }
  # cat("Summary statistics:\n")
  # cat("==================\n")
  # print(sumStats(x))
  # cat("Autocorrelation tests:\n")
  # cat("=====================\n")
  # nCoef = max(floor(nCoef), 1)
  # LB = ident(x, nCoef, nPar, TRUE)
  # print(LB)
  # pLB = plotAcfPacf(LB$SACF, LB$SPACF, s, length(x), TRUE)
  # cat("Gaussianity tests:\n")
  # cat("=================\n")
  # pGAUSS = gaussTest(x, TRUE)
  # cat("Ratio of variance tests:\n")
  # cat("=======================\n")
  # print(varTest(x, parts))
  # pCUSUM = cusum(x, TRUE)
  # Plotting
  p1 = autoplot(x)
  grid.arrange(
    p1, pLB[[1]], pLB[[2]], pGAUSS[[1]],
    pGAUSS[[2]], pCUSUM[[1]], pCUSUM[[2]],
    widths = c(1, 1),
    layout_matrix = rbind(c(1, 1),
                          c(1, 1),
                          c(2, 3),
                          c(4, 5),
                          c(6, 7))
  )
  return(out)
}
#' @title tsDisplay
#' @description Displays time series plot with autocorrelation functions
#'
#' @param y a vector, ts or tsibble object
#' @param nCoef number of autocorrelation coefficients to estimate
#' @param nPar number of parameters in a model if y is a residual
#' @param s seasonal period, number of observations per year
#'
#' @return No return value, called for side effects
#' 
#' @author Diego J. Pedregal
#'
#' @seealso \code{\link{colMedians}}, \code{\link{rowMedians}}, \code{\link{tests}},
#'          \code{\link{sumStats}}, \code{\link{gaussTest}}, \code{\link{ident}},
#'          \code{\link{cusum}}, \code{\link{varTest}}, \code{\link{conv}},
#'          \code{\link{armaFilter}}, \code{\link{dif}}, \code{\link{roots}},
#'          \code{\link{zplane}}, \code{\link{acft}}, \code{\link{slide}},
#'          \code{\link{plotSlide}}, \code{\link{Accuracy}},
#'          \code{\link{size}}
#'
#' @examples
#' tsDisplay(AirPassengers)
#' @rdname tsDisplay
#' @export
tsDisplay = function(y, nCoef = 25, nPar = 0, s = NA){
  # Plots time series and ACF and PACF
  if (is.list(y) && !is_tsibble(y)){
      x = as.ts(residuals(y))
      s = frequency(x)
  } else {
      x = as.ts(y)
  }
  LB = ident(x, nCoef, nPar, TRUE)
  pLB = plotAcfPacf(LB$SACF, LB$SPACF, frequency(x), length(x), TRUE)
  p1 = autoplot(x)
  grid.arrange(
    p1, pLB[[1]], pLB[[2]],
    widths = c(1, 1),
    layout_matrix = rbind(c(1, 1),
                          c(2, 3))
  )
}
#' @title sumStats
#' @description Summary statistics of a matrix of variables
#'
#' @details Position, dispersion, skewness, kurtosis, etc.
#'
#' @param y a vector, matrix of time series
#' @param decimals number of decimals for table
#'
#' @return Table of values in string matrix
#'
#' @author Diego J. Pedregal
#'
#' @seealso \code{\link{colMedians}}, \code{\link{rowMedians}}, \code{\link{tests}},
#'          \code{\link{gaussTest}}, \code{\link{ident}},
#'          \code{\link{cusum}}, \code{\link{varTest}}, \code{\link{conv}},
#'          \code{\link{armaFilter}}, \code{\link{dif}}, \code{\link{roots}},
#'          \code{\link{zplane}}, \code{\link{acft}}, \code{\link{slide}},
#'          \code{\link{plotSlide}}, \code{\link{Accuracy}}, \code{\link{tsDisplay}},
#'          \code{\link{size}}
#'
#' @examples
#' s <- sumStats(AirPassengers)
#' @rdname sumStats
#' @export
sumStats = function(y, decimals = 5){
  # Summary statistics of variables
  if (is.list(y) && !is_tsibble(y)){
      x = as.ts(residuals(y))
  } else {
      x = as.ts(y)
  }
  x = removeNaNs(x)
  q25 = quantile(x, 0.25, na.rm = TRUE)
  q75 = quantile(x, 0.75, na.rm = TRUE)
  maxx = max(x, na.rm = TRUE)
  minx = min(x, na.rm = TRUE)
  meanx = mean(x, na.rm = TRUE)
  sdx = sd(x, na.rm = TRUE)
  ind = !is.na(x)
  n = sum(ind)
  m1 = x[ind] - meanx
  m2 = sum(m1 ^ 2)
  skewness = (sum(m1 ^ 3) / n) / ((m2 / n) ^ (3 / 2))
  kurtosis = (sum(m1 ^ 4) / n) / ((m2 / n) ^ 2) - 3
  aux = c(length(y),
          sum(is.na(y)),
          minx,
          q25,
          meanx,
          2 * pt(-abs(meanx / (sdx / sqrt(n))), n - 1),
          median(x, na.rm = TRUE),
          q75,
          maxx,
          q75 - q25,
          maxx - minx,
          sdx,
          sdx * sdx,
          skewness,
          kurtosis)
  out = round(matrix(aux, 15, 1), decimals)
  colnames(out) = "Serie 1"
  rownames(out) = c("Data points: ",
                    "Missing: ",
                    "Minimum: ",
                    "1st quartile: ",
                    "Mean: ",
                    "P(Mean = 0): ",
                    "Median: ",
                    "3rd quartile:",
                    "Maximum: ",
                    "Interquartile range: ",
                    "Range: ",
                    "Satandard deviation: ",
                    "Variance: ",
                    "Skewness: ",
                    "Kurtosis: ")
  return(out)
}
#' @title gaussTest
#' @description Gaussianity tests
#'
#' @param y a vector, ts or tsibble object
#' @param runFromTests internal check
#'
#' @return No return value, called for side effects
#' 
#' @author Diego J. Pedregal
#'
#' @seealso \code{\link{colMedians}}, \code{\link{rowMedians}}, \code{\link{tests}},
#'          \code{\link{sumStats}}, \code{\link{ident}},
#'          \code{\link{cusum}}, \code{\link{varTest}}, \code{\link{conv}},
#'          \code{\link{armaFilter}}, \code{\link{dif}}, \code{\link{roots}},
#'          \code{\link{zplane}}, \code{\link{acft}}, \code{\link{slide}},
#'          \code{\link{plotSlide}}, \code{\link{Accuracy}}, \code{\link{tsDisplay}},
#'          \code{\link{size}}
#'
#' @examples
#' gaussTest(AirPassengers)
#' @rdname gaussTest
#' @export
gaussTest = function(y, runFromTests = FALSE){
  # Gaussianity test for independent time series
  if (is.list(y) && !is_tsibble(y)){
      x = as.ts(residuals(y))
  } else {
      x = as.ts(y)
  }
  x = x[!is.na(x)]
  n = length(x)
  df = data.frame(x)
  p1 = ggplot(df, aes(x = x)) +
          geom_histogram(aes(y = after_stat(density)),
                         bins = floor(sqrt(n)), color = "#000000", fill = "#0099F8") +
          geom_density(lwd = 1, color = "#000000", fill = "#F85700", alpha = 0.6) +
          stat_function(fun = dnorm, args = list(mean = mean(df$x), sd = sd(df$x)))
  # QQplot
  p2 = ggplot(df, aes(sample = x)) + stat_qq() + stat_qq_line(color = "darkred", linewidth = 1)
  shap = shapiro.test(x)
  if (runFromTests){
    return(list(p1, p2, shap))
  } else {
    grid.arrange(p1, p2, nrow = 2)
  }
}
#' @title ident
#' @description Autocorrelation functions of a time series
#'
#' @param y a vector, ts or tsibble object
#' @param nCoef number of autocorrelation coefficients to estimate
#' @param nPar number of parameters in a model if y is a residual
#' @param runFromTests internal check
#'
#' @return A vector with output table including ACF, etc.
#'
#' @author Diego J. Pedregal
#'
#' @seealso \code{\link{colMedians}}, \code{\link{rowMedians}}, \code{\link{tests}},
#'          \code{\link{sumStats}}, \code{\link{gaussTest}},
#'          \code{\link{cusum}}, \code{\link{varTest}}, \code{\link{conv}},
#'          \code{\link{armaFilter}}, \code{\link{dif}}, \code{\link{roots}},
#'          \code{\link{zplane}}, \code{\link{acft}}, \code{\link{slide}},
#'          \code{\link{plotSlide}}, \code{\link{Accuracy}}, \code{\link{tsDisplay}},
#'          \code{\link{size}}
#'
#' @examples
#' ident(AirPassengers)
#' @rdname ident
#' @export
ident = function(y,
                 nCoef = min(37, floor(length(y) / 4)),
                 nPar = 0,
                 runFromTests = FALSE){
  # ACF and PACF with Ljung Box tests and plots
  if (is.list(y) && !is_tsibble(y)){
      x = as.ts(residuals(y))
      s = frequency(x)
  } else {
      x = as.ts(y)
  }
  nCoef = max(nCoef, frequency(x))
  x = removeNaNs(x)
  x = x - mean(x, na.rm = TRUE)
  aux = matrix(0, nCoef + 1, 1)
  n = length(x)
  for (i in 0 : nCoef){
      prod = x[1 : (n - i)] * x[(i + 1) : n]
      ind = which(!is.na(prod))
      aux[i + 1, 1] = sum(prod[ind])
  }
  aux = aux / aux[1]
  ACF = as.numeric(tail(aux, nCoef))
  # PACF from ACF
  PACF <- ACF
  fi <- numeric(nCoef)
  fi[1] <- ACF[1]
  for (i in 1:(nCoef - 1)) {
      fi[i + 1] <- (ACF[i + 1] - sum(fi[1:i] * rev(ACF[1:i]))) / (1 - sum(fi * ACF[1:length(fi)]))
      PACF[i + 1] <- fi[i + 1]
      fi[1:i] <- fi[1:i] - fi[i + 1] * rev(fi[1:i])
  }
  #PACF = as.numeric(pacf(x, nCoef, plot = FALSE)$acf)
  BAND = rep(2 / sqrt(length(x)), nCoef)
  SIGa = rep(".", nCoef)
  SIGa[ACF > BAND] = "+"
  SIGa[ACF < -BAND] = "-"
  SIGp = matrix(".", nCoef)
  SIGp[PACF > BAND] = "+"
  SIGp[PACF < -BAND] = "-"
  BOX = rep(NA, nCoef)
  pval = BOX
  for (i in 1 : nCoef){
    ci = Box.test(x, i, "Ljung-Box", nPar)
    BOX[i] = ci$statistic
    pval[i] = ci$p.value
  }
  out = data.frame(SACF = round(ACF, 3), sa = SIGa,
                   LB = round(BOX, 3), p.val = round(pval, 3),
                   SPACF = round(PACF, 3), sp = SIGp)
  if (runFromTests){
    return(out)
  } else {
    plotAcfPacf(ACF, PACF, frequency(x), length(x))
  }
  return(out)
}
#' @title plotBar
#' @description Plot variable in bars
#'
#' @param ACF variable to plot
#' @param s seasonal period
#' @param n number of coefficients
#' @param label label for plot
#'
#' @return Handle of plot
#'
#' @author Diego J. Pedregal
#'
#' @rdname plotBar
#' @export
plotBar = function(ACF, s = 1, n = NA, label = "ACF"){
  # Plot of ACF & PACF
  ncoef = length(ACF)
  band = 2 / sqrt(n)
  ACFs = rep(0, ncoef)
  ACFs[seq(s, ncoef, s)] = ACF[seq(s, ncoef, s)]
  df = data.frame(lag = seq(1, length(ACF)), ACF, ACFs)
  p = ggplot(df, aes(x = lag, y = ACF)) +
    geom_hline(aes(yintercept = 0)) +
    geom_segment(aes(xend = lag, yend = 0)) +
    labs(y = label) + theme(legend.position = "none")
  if (!is.na(n)){
    p = p +
      geom_hline(aes(yintercept = band), linetype = 2, color = 'red') +
      geom_hline(aes(yintercept = -band), linetype = 2, color = 'red')
  }
  if (s > 1){
    p = p + geom_segment(data = df,
                         aes(y = ACFs, xend = lag, yend = 0, color = "red"))
  }
  return(p)
}
#' @title plotAcfPacf
#' @description Plot of ACF and PACF
#'
#' @param ACF variable to plot
#' @param PACF second variable to plot
#' @param s seasonal period
#' @param n number of coefficients
#' @param runFromTest internal check variable
#'
#' @return No return value, called for side effects
#' 
#' @author Diego J. Pedregal
#'
#' @rdname plotAcfPacf
#' @export
plotAcfPacf = function(ACF, PACF, s = 1, n = NA, runFromTest = FALSE){
  # Plot of ACF & PACF
  p1 = plotBar(ACF, s, n)
  p2 = plotBar(PACF, s, n, "PACF")
  if (runFromTest){
    return(list(p1, p2))
  } else {
      grid.arrange(p1, p2, nrow = 2)
  }
}
#' @title cusum
#' @description Cusum and cusumsq tests
#'
#' @param y a vector, ts or tsibble object
#' @param runFromTest internal check variable
#'
#' @return No return value, called for side effects
#' 
#' @author Diego J. Pedregal
#'
#' @seealso \code{\link{colMedians}}, \code{\link{rowMedians}}, \code{\link{tests}},
#'          \code{\link{sumStats}}, \code{\link{gaussTest}}, \code{\link{ident}},
#'          \code{\link{varTest}}, \code{\link{conv}},
#'          \code{\link{armaFilter}}, \code{\link{dif}}, \code{\link{roots}},
#'          \code{\link{zplane}}, \code{\link{acft}}, \code{\link{slide}},
#'          \code{\link{plotSlide}}, \code{\link{Accuracy}}, \code{\link{tsDisplay}},
#'          \code{\link{size}}
#'
#' @examples
#' cusum(AirPassengers)
#' @rdname cusum
#' @export
cusum = function(y, runFromTest = FALSE){
  # CUSUM and CUSUMsq tests
  if (is.list(y) && !is_tsibble(y)){
      x = as.ts(residuals(y))
  } else {
      x = as.ts(y)
  }
  x = removeNaNs(x)
  n = length(x)
  mz = mean(x, na.rm = TRUE)
  stdz = sd(x, na.rm = TRUE)
  NMiss = !is.na(x)
  Miss = is.na(x)
  standz = matrix(NA, n, 1)
  cusumLine = standz
  cusumsq = standz
  NMissi = which(NMiss)
  zmi = x[NMissi] - mz
  standz[NMissi] = zmi / stdz
  cusumLine[NMissi] = cumsum(standz[NMissi])
  cusumsq[NMissi] = cumsum(zmi ^ 2 / (sum(zmi ^ 2)))
  # Calculating bands
  t = 1 : n
  bands= matrix(NA, n, 2)
  bands[, 1] = c(0.948 * sqrt(n) + 2 * 0.948 * t / sqrt(n))
  bands[, 2]= -bands[, 1]
  # Plotting CUSUM
  miny = min(min(bands, na.rm = TRUE), min(cusumLine, na.rm = TRUE))
  maxy = max(max(bands, na.rm = TRUE), max(cusumLine, na.rm = TRUE))
  p1 = autoplot(as.ts(cusumLine)) +
    autolayer(as.ts(bands[, 1]), color = "red", linetype = 2) +
    autolayer(as.ts(bands[, 2]), color = "red", linetype = 2) +
    autolayer(as.ts(rep(0, length(cusumLine))), color = "red") +
    labs(y = "CUSUM")
  # Plotting CUSUMsq
  wu = .948 * sqrt(n) + 2 * .948 * t / sqrt(n)
  bands = cbind(-.32894 + t / n, .32894 + t / n)
  miny = min(min(bands, na.rm = TRUE), min(cusumsq, na.rm = TRUE))
  maxy = max(max(bands, na.rm = TRUE), max(cusumsq, na.rm = TRUE))
  p2 = autoplot(as.ts(cusumsq)) +
    autolayer(as.ts(bands[, 1]), color = "red", linetype = 2) +
    autolayer(as.ts(bands[, 2]), color = "red", linetype = 2) +
    autolayer(as.ts(t / length(cusumLine)), color = "red") +
    labs(y = "CUSUMsq")
  if (runFromTest){
    return(list(p1, p2))
  } else {
    grid.arrange(p1, p2, nrow = 2)
  }
}
#' @title varTest
#' @description Ratio of variances test
#'
#' @param y a vector, ts or tsibble object
#' @param parts portion of sample to estimate variances
#'
#' @return Table with test results
#'
#' @author Diego J. Pedregal
#'
#' @seealso \code{\link{colMedians}}, \code{\link{rowMedians}}, \code{\link{tests}},
#'          \code{\link{sumStats}}, \code{\link{gaussTest}}, \code{\link{ident}},
#'          \code{\link{cusum}}, \code{\link{conv}},
#'          \code{\link{armaFilter}}, \code{\link{dif}}, \code{\link{roots}},
#'          \code{\link{zplane}}, \code{\link{acft}}, \code{\link{slide}},
#'          \code{\link{plotSlide}}, \code{\link{Accuracy}}, \code{\link{tsDisplay}},
#'          \code{\link{size}}
#'
#' @examples
#' varTest(AirPassengers)
#' @rdname varTest
#' @export
varTest = function(y, parts = 1 / 3){
  # Ratio of varainces tests
  if (is.list(y) && !is_tsibble(y)){
      x = as.ts(residuals(y))
  } else {
      x = as.ts(y)
  }
  x = removeNaNs(x)
  n = length(x)
  n1 = floor(n * parts)
  aux = var.test(head(x, n1), tail(x, n1))
  out = data.frame(Portion_of_data = paste(round(parts, 5)),
                   F_statistic = round(aux$statistic, 4),
                   p.value = round(aux$p.value, 4))
  rownames(out) = ""
  return(out)
}
#' @title conv
#' @description 1D convolution: filtering or polynomial multiplication
#'
#' @param ... list of vectors to convolute
#'
#' @return Convolution of all input vectors
#'
#' @author Diego J. Pedregal
#'
#' @seealso \code{\link{colMedians}}, \code{\link{rowMedians}}, \code{\link{tests}},
#'          \code{\link{sumStats}}, \code{\link{gaussTest}}, \code{\link{ident}},
#'          \code{\link{cusum}}, \code{\link{varTest}},
#'          \code{\link{armaFilter}}, \code{\link{dif}}, \code{\link{roots}},
#'          \code{\link{zplane}}, \code{\link{acft}}, \code{\link{slide}},
#'          \code{\link{plotSlide}}, \code{\link{Accuracy}}, \code{\link{tsDisplay}},
#'          \code{\link{size}}
#'
#' @examples
#' conv(c(1, -1), c(1, -2, 1))
#' conv(c(1, -1), c(1, 0.8))
#' @rdname conv
#' @export
conv = function(...) {
  # 1D convolution: filtering or polynomial multiplication
  args = list(...)
  n = ...length()
  if (n > 2){
    return(conv(args[[1]], do.call(conv, args[2 : n])))
  } else {
    # Convolution of n vectors
    #This function convolves two vectors x and y
    #x and y must be numeric vectors, length at least 1
    #z=x*y is returned
    #L. Schweitzer A4, 1
    lx = length(args[[1]])
    ly = length(args[[2]])
    lz = lx + ly - 1
    z = 0 * numeric(lz)
    for (i in 1 : lx) {
      # begin convolving x and y
      for (j in 1 : ly) {
        z[(i - 1) + (j - 1) + 1] = z[(i - 1) + (j - 1) + 1] + args[[1]][i] * args[[2]][j]
      }
    } # end convolving x and y
    return(z)
  }
}

#' @title armaFilter
#' @description Filter of time series
#'
#' @param MA numerator polynomial
#' @param AR denominator polynomial
#' @param y a vector, ts or tsibble object
#'
#' @return Filtered time series
#'
#' @author Diego J. Pedregal
#'
#' @seealso \code{\link{colMedians}}, \code{\link{rowMedians}}, \code{\link{tests}},
#'          \code{\link{sumStats}}, \code{\link{gaussTest}}, \code{\link{ident}},
#'          \code{\link{cusum}}, \code{\link{varTest}}, \code{\link{conv}},
#'          \code{\link{dif}}, \code{\link{roots}},
#'          \code{\link{zplane}}, \code{\link{acft}}, \code{\link{slide}},
#'          \code{\link{plotSlide}}, \code{\link{Accuracy}}, \code{\link{tsDisplay}},
#'          \code{\link{size}}
#'
#' @examples
#' y <- armaFilter(1, c(1 , -0.8), rnorm(200))
#' @rdname armaFilter
#' @export
armaFilter = function(MA = 1, AR = 1, y){
  # Filtering a time series with linear filter
  xs = as.ts(y)
  q = length(MA)
  p = length(AR)
  x = conv(xs, MA)
  if (p > 1){
    x = stats::filter(x, -AR[2 : length(AR)], "recursive")
    x = ts(x[1 : length(xs)],
           frequency = frequency(xs),
           start = time(xs)[1])
  } else {
    x = ts(x[length(MA) : length(xs)],
           frequency = frequency(xs),
           start = time(xs)[length(MA)])
  }
  if (is_tsibble(x)){
    x = as_tsibble(x)
  }
  return(x)
}
#' @title dif
#' @description Discrete differencing of time series
#'
#' @param y a vector, ts or tsibble object
#' @param difs vector with differencing orders
#' @param seas vector of seasonal periods
#'
#' @return Differenced time series
#'
#' @author Diego J. Pedregal
#'
#' @seealso \code{\link{colMedians}}, \code{\link{rowMedians}}, \code{\link{tests}},
#'          \code{\link{sumStats}}, \code{\link{gaussTest}}, \code{\link{ident}},
#'          \code{\link{cusum}}, \code{\link{varTest}}, \code{\link{conv}},
#'          \code{\link{armaFilter}}, \code{\link{roots}},
#'          \code{\link{zplane}}, \code{\link{acft}}, \code{\link{slide}},
#'          \code{\link{plotSlide}}, \code{\link{Accuracy}}, \code{\link{tsDisplay}},
#'          \code{\link{size}}
#'
#' @examples
#' dif(AirPassengers)
#' dif(AirPassengers, 2)
#' dif(AirPassengers, c(1, 1), c(1, 12))
#' @rdname dif
#' @export
dif = function(y, difs = 1, seas = 1){
  # Mixture of differences operating on x
  x = as.ts(y)
  n = length(difs)
  pol = 1
  for (i in 1 : n){
    poli = c(1, rep(0, seas[i] - 1), -1)
    if (difs[i] > 0){
      for (j in 1 : difs[i]){
        pol = conv(pol, poli)
      }
    }
  }
  dx = armaFilter(pol, 1, x)
  if (is_tsibble(x)){
    dx = as_tsibble(dx)
  }
  return(head(dx, (length(x) - length(pol) + 1)))
}
#' @title roots
#' @description Roots of polynomial
#'
#' @param x coefficients of polynomial in descending order
#'
#' @return Roots of polynomial
#'
#' @author Diego J. Pedregal
#'
#' @seealso \code{\link{colMedians}}, \code{\link{rowMedians}}, \code{\link{tests}},
#'          \code{\link{sumStats}}, \code{\link{gaussTest}}, \code{\link{ident}},
#'          \code{\link{cusum}}, \code{\link{varTest}}, \code{\link{conv}},
#'          \code{\link{armaFilter}}, \code{\link{dif}},
#'          \code{\link{zplane}}, \code{\link{acft}}, \code{\link{slide}},
#'          \code{\link{plotSlide}}, \code{\link{Accuracy}}, \code{\link{tsDisplay}},
#'          \code{\link{size}}
#'
#' @examples
#' roots(c(1, -2 ,1))
#' roots(conv(c(1, -1), c(1, 0.8)))
#' @rdname roots
#' @export
roots = function(x){
  # Roots of a polynomial
  if (length(x) > 1){
    return(polyroot(rev(x)))
  } else {
    return(NULL)
  }
}
#' @title zplane
#' @description Real-imaginary plane to show roots of digital filters (ARMA)
#'
#' @details Shows the real-imaginary plane to show zeros (roots of numerator or
#' MA polynomial) and poles (roots of denominator of AR polynomial). Unit roots
#' and real vs imaginary roots can be seen by eye
#'
#' @param MApoly coefficients of numerator polynomial in descending order
#' @param ARpoly coefficients of denominator polynomial in descending order
#'
#' @return No return value, called for side effects
#' 
#' @author Diego J. Pedregal
#'
#' @seealso \code{\link{colMedians}}, \code{\link{rowMedians}}, \code{\link{tests}},
#'          \code{\link{sumStats}}, \code{\link{gaussTest}}, \code{\link{ident}},
#'          \code{\link{cusum}}, \code{\link{varTest}}, \code{\link{conv}},
#'          \code{\link{armaFilter}}, \code{\link{dif}}, \code{\link{roots}},
#'          \code{\link{acft}}, \code{\link{slide}},
#'          \code{\link{plotSlide}}, \code{\link{Accuracy}}, \code{\link{tsDisplay}},
#'          \code{\link{size}}
#' @examples
#' zplane(c(1, -2, 1), c(1, -0.8))
#' @rdname zplane
#' @export
zplane = function(MApoly = 1, ARpoly = 1){
  # zplane plot
  # if (is.list(MApoly)){   # MApoly is an ARIMA fable model
  #     aux = glance(MApoly)
  #     arRoots = aux$ar_roots[[1]]
  #     maRoots = aux$ma_roots[[1]]
  #     modN = abs(arRoots) ^ 2
  #     arRoots = complex(real = Re(arRoots) / modN, imaginary = -Im(arRoots) / modN)
  #     modN = abs(maRoots) ^ 2
  #     maRoots = complex(real = Re(maRoots) / modN, imaginary = -Im(maRoots) / modN)
  # } else {
      arRoots = roots(ARpoly)
      maRoots = roots(MApoly)
  # }
  reaux = NA
  imaux = NA
  x = seq(-1, 1, 0.005)
  y = sqrt(1 - x ^ 2)
  # Dibujando círculo
  p = ggplot() +
    geom_circle(aes(x0 = 0, y0 = 0, r = 1)) +
    geom_hline(yintercept=0, linetype="dashed") +
    geom_vline(xintercept=0, linetype="dashed") +
    coord_fixed() +
    labs(x = "Real part", y = "Imaginary part") +
    ggtitle("Poles (x - AR) and zeros (o - MA)")
  # Añadiendo raices
  if (length(arRoots) > 0){
    modAR = abs(arRoots)
    # Stationary
    aux = arRoots[modAR < 0.99]
    aux = data.frame(reaux = Re(aux), imaux = Im(aux))
    if (size(aux)[1] > 0){
      p = p + geom_point(data = aux, aes(x = reaux, y = imaux),
                         shape = "x", size = 4)
    }
    # Non-stationary
    aux = arRoots[modAR >= 0.99]
    aux = data.frame(reaux = Re(aux), imaux = Im(aux))
    if (size(aux)[1] > 0){
      p = p + geom_point(data = aux, aes(x = reaux, y = imaux),
                         shape = "x", size = 4, color = "red")
    }
  }
  if (length(maRoots) > 0){
    # Inverible
    modMA = abs(maRoots)
    aux = maRoots[modMA < 0.99]
    aux = data.frame(reaux = Re(aux), imaux = Im(aux))
    if (size(aux)[1] > 0){
      p = p + geom_point(data = aux, aes(x = reaux, y = imaux),
                         shape = "o", size = 4)
    }
    # Non-invertible
    aux = maRoots[modMA >= 0.99]
    aux = data.frame(reaux = Re(aux), imaux = Im(aux))
    if (size(aux)[1] > 0){
      p = p + geom_point(data = aux, aes(x = reaux, y = imaux),
                         shape = "o", size = 4, color = "red")
    }
  }
  print(p)
}
#' @title arma2tsi
#' @description AR polynomial coefficients of ARMA model
#'
#' @param MApoly coefficients of numerator polynomial in descending order
#' @param ARpoly coefficients of denominator polynomial in descending order
#' @param n number of coefficients
#'
#' @return Tsi (MA form) coefficients of equivalent ARMA model
#' 
#' @author Diego J. Pedregal
#'
#' @rdname arma2tsi
#' @export
arma2tsi = function(MApoly, ARpoly, n = 100){
  # Converts ARMA to pure AR
  return(armaFilter(MApoly, ARpoly, c(1, rep(0, n - 1))))
}
#' @title acft
#' @description Theoretical autocorrelation functions of ARMA models
#'
#' @param MApoly coefficients of numerator polynomial in descending order
#' @param ARpoly coefficients of denominator polynomial in descending order
#' @param ncoef number of coefficients
#' @param s seasonal period, number of observations per year
#'
#' @return Theoretical autocorrelation functions
#'
#' @author Diego J. Pedregal
#'
#' @seealso \code{\link{colMedians}}, \code{\link{rowMedians}}, \code{\link{tests}},
#'          \code{\link{sumStats}}, \code{\link{gaussTest}}, \code{\link{ident}},
#'          \code{\link{cusum}}, \code{\link{varTest}}, \code{\link{conv}},
#'          \code{\link{armaFilter}}, \code{\link{dif}}, \code{\link{roots}},
#'          \code{\link{zplane}}, \code{\link{slide}},
#'          \code{\link{plotSlide}}, \code{\link{Accuracy}}, \code{\link{tsDisplay}},
#'          \code{\link{size}}
#'
#' @examples
#' acft(c(1, -0.8), c(1, 0.8))
#' @rdname acft
#' @export
acft = function(MApoly = 1, ARpoly = 1, ncoef = 38, s = 1){
  # Theoretical ACF and PACF from ARMA model
  correct = TRUE
  if (length(ARpoly) > 1 && any(abs(roots(ARpoly)) >= 1)){
    warning('Non-stationary model')
    correct = FALSE
  }
  if (length(MApoly) > 1 && any(abs(roots(MApoly)) >= 1)){
    warning('Non-invertible model')
    correct = FALSE
  }
  if (!correct){
    stop('SACF and SPACF do not exist for this model!!!')
    return()
  }
  p = size(ARpoly) - 1
  q = size(MApoly) - 1
  out = matrix(1, ncoef, 3)
  out[, 1] = 1 : ncoef
  # ACF
  aux = 250
  MAPi = rep(1, aux)
  MAP = MApoly
  if (p > 0){
    while (MAPi[aux] > 1e-4){
      aux = aux + 100
      MAPi = arma2tsi(MApoly, ARpoly, aux)
    }
    MAP = MAPi
  }
  if (size(MAP) < aux){
    MAP = c(MAP, rep(0, aux - size(MAP)))
  }
  gam0= sum(MAP ^ 2)
  for (i in 1 : ncoef){
    out[i, 2] = sum(MAP[(i + 1) : aux] * MAP[1 : (aux - i)]) / gam0
  }
  # PACF
  fi = rep(0, ncoef)
  fi[1] = out[1, 2]
  out[1, 3] = fi[1]
  for (p in 1 : (ncoef - 1)){
    fi[p + 1] = (out[p + 1, 2] - sum(fi[1 : p] * rev(out[1 : p, 2]))) / (1 - sum(fi * out[, 2]))
    out[p + 1, 3] = fi[p + 1]
    fi[1 : p] = fi[1 : p] - fi[p + 1] * rev(fi[1 : p])
  }
  plotAcfPacf(out[, 2], out[, 3], s)
  colnames(out) = c("lag", "SACF", "SPACF")
  return(out)
}
#' @title slide
#' @description Rolling forecasting of a matrix of time series
#'
#' @details Takes time series and run forecasting methods implemented in function
#' forecFun h steps ahead along the time series y, starting at forecasting
#' origin orig, and moving step observations ahead. Forecasts may be run in parallel
#' by setting parallel to TRUE. A fixed window width may be
#' specified with input window. The output is of dimensions (h, nOrigs, nModels, nSeries)
#'
#' @param y a vector, a matrix or a list of time series
#' @param orig starting forecasting origin
#' @param forecFun user function that implements forecasting methods
#' @param ... rest of inputs to forecFun function
#' @param h forecasting horizon
#' @param step observations ahead to move the forecasting origin
#' @param output output TRUE/FALSE
#' @param window fixed window width in number of observations (NA for non fixed)
#' @param parallel run forecasts in parallel
#'
#' @return An array of forecasts of dimensions (horizon x nOrigs x nModels x nSeries)
#'
#' @author Diego J. Pedregal
#'
#' @seealso \code{\link{colMedians}}, \code{\link{rowMedians}}, \code{\link{tests}},
#'          \code{\link{sumStats}}, \code{\link{gaussTest}}, \code{\link{ident}},
#'          \code{\link{cusum}}, \code{\link{varTest}}, \code{\link{conv}},
#'          \code{\link{armaFilter}}, \code{\link{dif}}, \code{\link{roots}},
#'          \code{\link{zplane}}, \code{\link{acft}},
#'          \code{\link{plotSlide}}, \code{\link{Accuracy}}, \code{\link{tsDisplay}},
#'          \code{\link{size}}
#'
#' @examples
#' \dontrun{slide(AirPassengers, 100, forecFun)}
#' @rdname slide
#' @export
slide <- function(y,
                  orig,
                  forecFun,
                  ...,
                  h = 12,
                  step = 1,
                  output = TRUE,
                  window = NA,
                  parallel = FALSE) {
  # suppressWarnings()
  # Preparación de datos
  yisList <- is.list(y)
  yisListList <- if (yisList) is.list(y[[1]]) else FALSE
  if (!yisList) {
    if (!is.ts(y)) y <- as.ts(y)
    if (length(size(y)) == 1) {
      nSeries <- 1
      y <- list(list(y))
    } else {
      nSeries <- ncol(y)
      aux <- frequency(y)
      auxs <- start(y)
      y <- split(y, col(y))
      y <- lapply(y, ts, start = auxs, frequency = aux)
      y <- lapply(y, function(x) list(x))
    }
  } else if (!yisListList) {
    nSeries <- length(y)
    y <- lapply(y, function(x) list(x))
  } else {
    nSeries <- length(y)
  }
  # Detección de tipo de entrada para forecFun
  test1 <- !is.null(tryCatch(forecFun(y[[1]], h, ...), error = function(e) NULL))
  test2 <- !is.null(tryCatch(forecFun(y[[1]][[1]], h, ...), error = function(e) NULL))
  isList <- if (test1 && !test2) TRUE else if (!test1 && test2) FALSE else yisList
  # Ejecución
  if (nSeries == 1) {
    listOut <- lapply(y, slideAux, orig, forecFun, h, step, output, FALSE, window, parallel, isList, ...)
  } else {
    if (parallel) {
      cores <- max(1, parallel::detectCores() - 1)
      cl <- parallel::makeCluster(cores)
      on.exit(parallel::stopCluster(cl))
      parallel::clusterExport(cl, varlist = c("slideAux", "orig", "forecFun", "h", "step", "output", "window", "isList"), envir = environment())
      listOut <- parallel::parLapply(cl, y, function(x) slideAux(x, orig, forecFun, h, step, output, FALSE, window, FALSE, isList, ...))
    } else {
      listOut <- lapply(y, slideAux, orig, forecFun, h, step, output, FALSE, window, FALSE, isList, ...)
    }
  }
  # Construcción del array de salida
  out <- array(NA, c(dim(listOut[[1]]), nSeries))
  for (i in 1:nSeries) {
    out[, , , i] <- listOut[[i]]
  }
  dimnames(out)[[3]] <- dimnames(listOut[[1]])[[3]]
  # options(warn = 0)
  return(out)
}
#' @title slideAux
#' @description Auxiliary function run from slide
#'
#' @param y a vector or matrix of time series
#' @param orig starting forecasting origin
#' @param forecFun user function that implements forecasting methods
#' @param h forecasting horizon
#' @param step observations ahead to move the forecasting origin
#' @param output output TRUE/FALSE
#' @param graph fraphical output TRUE/FALSE
#' @param window fixed window width in number of observations (NA for non fixed)
#' @param parallel run forecasts in parallel
#' @param isList whether the input data y is a list or a matrix
#' @param ... rest of inputs to forecFun function
#'
#' @return Auxiliary output of slide function for just one time series
#' 
#' @author Diego J. Pedregal
#'
#' @rdname slideAux
#' @export
slideAux = function(y,
                    orig,
                    forecFun,
                    h = 12,
                    step = 1,
                    output = TRUE,
                    graph = TRUE,
                    window = NA,
                    parallel = FALSE,
                    isList = FALSE,
                    ...){
  # Rolling for 1 series
  # out = [h, nOrigs, nModels]
  n = length(y[[1]])
  origs = seq(orig, n - h, step)
  nOr = length(origs)
  # # Detecting whether forecFun accepts list as input
  # isList = FALSE
  # x = names(formals(forecFun))[1]
  # code <- deparse(body(forecFun))
  # firstTest = any(grepl(paste0(x, "\\$"), code))
  # secondTest = any(grepl(paste0(x, "\\[\\["), code))
  # if (firstTest || secondTest){
  #     isList = TRUE
  # }
  # Checking kind of input to function forecFun
  # outi = tryCatch({
  #     forecFun(y, h, ...)  # llamada a la función
  # }, error = function(e) {
  #     return(NULL)  # puedes devolver un valor por defecto si lo deseas
  # })
  # isList = TRUE
  # if (is.null(outi)) {
  #     outi = forecFun(y[[1]], h, ...)
  #     isList = FALSE
  # }
  if (isList) {
    outi = forecFun(y, h, ...)
  } else {
    outi = forecFun(y[[1]], h, ...)
  }
  outi = as.matrix(outi)
  nMethods = dim(outi)[2]
  out = array(NA, c(h, length(origs), nMethods))
  out[, 1, ] = outi
  dimnames(out)[[3]] = colnames(outi)
  dataList = vector(mode = "list", length = nOr - 1)
  if (nOr > 1){
    for (j in 1 : (nOr - 1)){
      if (isList) {
        dataList[[j]] = y
        if (is.na(window)){
          dataList[[j]][[1]] = subset(y[[1]], end = orig + j)
        } else {
          dataList[[j]][[1]] = subset(y[[1]], start = orig - window + step, end = orig + j)
        }
      } else {
        if (is.na(window)){
          dataList[[j]] = subset(y[[1]], end = orig + j)
        } else {
          dataList[[j]] = subset(y[[1]], start = orig - window + step, end = orig + j)
        }
      }
    }
    
  }
  # if (parallel){
  #         listOut = mclapply(dataList, forecFun, h, ..., mc.cores = detectCores())
  #         # if (substr(.Platform$OS.type, 1, 1) == "w"){
  #         #     listOut = parallelsugar::mclapply(dataList, forecFun, h, ..., mc.cores = detectCores())
  #         # } else {
  #         #     listOut = parallel::mclapply(dataList, forecFun, h, ..., mc.cores = detectCores())
  #         # }
  #         
  # } else {
  listOut = lapply(dataList, forecFun, h, ...)
  # }
  if (nOr > 1){
    for (i in 2 : nOr){
      out[, i, ] = as.matrix(listOut[[i - 1]])
    }
  }
  return(out)
}
#' @title plotSlide
#' @description Plot summarised results from slide
#'
#' @param py1 output from slide function
#' @param y a vector, matrix or list of time series (the same used in slide call)
#' @param orig starting forecasting origin (the same used in slide call)
#' @param step observations ahead to move the forecasting origin (the same used in slide call)
#' @param errorFun user function to calculate error measures
#' @param collectFun aggregation function (mean, median, etc.)
#'
#' @return An array of forecasting errors of dimensions (horizon x nOrigs x nModels x nSeries)
#'
#' @author Diego J. Pedregal
#'
#' @seealso \code{\link{colMedians}}, \code{\link{rowMedians}}, \code{\link{tests}},
#'          \code{\link{sumStats}}, \code{\link{gaussTest}}, \code{\link{ident}},
#'          \code{\link{cusum}}, \code{\link{varTest}}, \code{\link{conv}},
#'          \code{\link{armaFilter}}, \code{\link{dif}}, \code{\link{roots}},
#'          \code{\link{zplane}}, \code{\link{acft}}, \code{\link{slide}},
#'          \code{\link{Accuracy}}, \code{\link{tsDisplay}},
#'          \code{\link{size}}
#'
#' @examples
#' \dontrun{plotSlide(py1, AirPassengers, 100, 1, errorFun)}
#' @rdname plotSlide
#' @export
plotSlide = function(py1, y, orig, step = 1,
                     errorFun,
                     collectFun = mean){
        # py1 = [h, nOrigs, nModels, nSeries]
        h = dim(py1)[1]
        nOrigs = dim(py1)[2]
        nMethods = dim(py1)[3]
        nSeries = dim(py1)[4]
        if (is.na(nSeries)){
                nSeries = 1
                py = array(NA, c(dim(py1), 1))
                dimnames(py) = dimnames(py1)
                py[, , , 1] = py1
        } else {
                py = py1
        }
        metrics = matrix(NA, h, nMethods)
        colnames(metrics) = dimnames(py)[[3]]
        outj = array(NA, c(h, nOrigs, nMethods, nSeries))
        # Converting list of lists to just a list of time series
        yisList = is.list(y)
        yisListList = FALSE
        if (yisList)
                yisListList = is.list(y[[1]])
        if (yisListList) {
                y = lapply(y, function(x) x[[1]])
        }
        # End of converting list of lists
        for (i in 1 : nOrigs){
                if (yisList) {
                        actuali = lapply(y, function(serie) {
                                subset(serie, end = orig + (i - 1) * step + h)
                        })
                } else {
                        actuali = subset(y, end = orig + (i - 1) * step + h)
                }
                aux = array(NA, c(h, nMethods, nSeries))
                aux[, , ] = py[, i, , ]
                colnames(aux) = dimnames(py)[[3]]
                if (nSeries == 1){
                        for (j in 1 : nMethods){
                                outj[, i, j, 1] = errorFun(aux[, j, 1], actuali)
                        }
                } else {
                        for (k in 1 : nSeries){
                                for (j in 1 : nMethods){
                                        if (yisList) {
                                                auxx = as.vector(actuali[[k]])
                                        } else {
                                                auxx = as.vector(actuali[, k])
                                        }
                                        outj[, i, j, k] = errorFun(aux[, j, k], auxx)
                                }
                        }
                }
                # outj[, i, , ] = AccuracyAll(aux, actuali, collectFun = collectFun, one = FALSE, )[, , criteriaColumn]
        }
        for (m in 1 : nMethods){
                for (j in 1 : h){
                        metrics[j, m] = collectFun(outj[j, , m, ], na.rm = TRUE)
                }
        }
        plotH = autoplot(ts(metrics, frequency = 1, start = 1), ylab = "")
        print(plotH)
        return(outj)
}
#' @title Accuracy
#' @description Accuracy for 1 time series y and several forecasting
#'  methods py and h steps ahead py is h x nMethods x nSeries
#'
#' @param py matrix of forecasts (h x nMethods x nForecasts)
#' @param y a matrix of actual values (n x nForecasts)
#' @param s seasonal period, number of observations per year
#' @param collectFun aggregation function (mean, median, etc.)
#'
#' @return Table of accuracy results
#'
#' @author Diego J. Pedregal
#'
#' @seealso \code{\link{colMedians}}, \code{\link{rowMedians}}, \code{\link{tests}},
#'          \code{\link{sumStats}}, \code{\link{gaussTest}}, \code{\link{ident}},
#'          \code{\link{cusum}}, \code{\link{varTest}}, \code{\link{conv}},
#'          \code{\link{armaFilter}}, \code{\link{dif}}, \code{\link{roots}},
#'          \code{\link{zplane}}, \code{\link{acft}}, \code{\link{slide}},
#'          \code{\link{plotSlide}}, \code{\link{tsDisplay}},
#'          \code{\link{size}}
#'
#' @examples
#' \dontrun{Accuracy(py, y, 12)}
#' @rdname Accuracy
#' @export
Accuracy = function(py, y, s = frequency(y), collectFun = mean){
    # Accuracy for 1 time series y and several forecasting methods py and h steps ahead
    # py is h x nMethods x nSeries
    # y  is n x nSeries
    spy = length(size(py))
    if (is.ts(py) && is.ts(y)){
        tpy = time(py)
        ty = time(y)
        # if (min(tpy) > max(ty)){
        if (min(tpy) - max(ty) > 1e-5){
            stop("No errors to estimate, check dates!!!")
        }
        # if (max(ty) < max(tpy)){
        if (max(tpy) - max(ty) > 1e-5){
            # Cut forecasts
            # py = head(py, which(max(ty) == tpy))
            py = head(py, which.min(abs(max(ty) - tpy)))
        }
        # if (max(tpy) < max(ty)){
        if (max(ty) - max(tpy) > 1e-5){
            # Cut time series
            # y = head(y, which(max(tpy) == ty))
            y = head(y, which.min(abs(max(tpy) - ty)))
        }
    }
    if (spy == 1){
        stop("Input \'py\' should be: h x number of methods x number of series!!!")
    } else if (spy == 2){
        aux = array(NA, c(dim(py), 1))
        aux[, , ] = as.matrix(py)
        dimnames(aux)[[2]] = colnames(py)
        py = aux
        y = as.ts(y)
    }
    h = dim(py)[1]
    ny = dim(py)[2]
    nSeries = dim(py)[3]
    n = size(y)[1]
    if (nSeries == 1){
        y = ts(matrix(y, length(y), 1), start = start(y), frequency = frequency(y))
    }
    insample = FALSE
    if (n > h){
        insample = TRUE
    } else if (n < h){
        stop("Insample data should be longer than forecasting horizon!!!")
    }
    out = matrix(NA, ny, 7 + 3 * insample)
    rownames(out) = c(1 : ny)
    namesc = c("ME", "RMSE", "MAE", "MPE", "PRMSE", "MAPE", "sMAPE")
    if (insample){
        namesc = c(namesc, "MASE", "RelMAE", "Theil\'s U")
    }
    colnames(out) = namesc
    rownames(out) = colnames(py)
    for (i in 1 : ny){
        e = array(py[, i, ], c(h, nSeries)) - tail(y, h)
        p = 100 * e / tail(y, h)
        aux = cbind(colMeans(e, na.rm = TRUE),
                    sqrt(colMeans(e * e, na.rm = TRUE)),
                    colMeans(abs(e), na.rm = TRUE),
                    colMeans(p, na.rm = TRUE),
                    sqrt(colMeans(p * p, na.rm = TRUE)),
                    colMeans(abs(p), na.rm = TRUE),
                    colMeans(200 * abs(e) / (array(py[, i, ], c(h, nSeries)) + tail(y, h)), na.rm = TRUE))
        if (insample){
            theil = 100 * (tail(y, h) - y[(n - h) : (n - 1), ]) / tail(y, h)
            fRW = y[(s + 1) : (n - h), ] - y[1 : (n - h - s), ]
            if (nSeries == 1){
                fRW = matrix(fRW, length(fRW), 1)
            }
            aux = cbind(aux,
                        colMeans(abs(e), na.rm = TRUE) / colMeans(abs(fRW), na.rm = TRUE),
                        colSums(abs(e), na.rm = TRUE) / colSums(abs(fRW), na.rm = TRUE),
                        sqrt(colSums(p * p, na.rm = TRUE) / colSums(theil * theil, na.rm = TRUE)))
        }
        out[i, ] = apply(aux, MARGIN = 2, FUN = collectFun, na.rm = TRUE)
    }
    return(out)
}
#' @title box.cox
#' @description Runs Box-Cox transform of a time series
#'
#' @param x Time series object.
#' @param lambda Lambda parameter for Box-Cox transform.
#' 
#' @return Box-Cox transformed time series
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{inv.box.cox}}, \code{\link{UC}}, \code{\link{UCforecast}}, \code{\link{UCvalidate}}, \code{\link{UCfilter}}, \code{\link{UCsmooth}}, 
#'          \code{\link{UCdisturb}}, \code{\link{UCcomponents}}
#'          
#' @examples
#' \donttest{
#' y <- box.cox(AirPassengers, 0.5)
#' plot(y)
#' }
#' @rdname box.cox
#' @export
box.cox <- function(x, lambda){
        if (abs(lambda) < 1e-4){
                return(log(x))
        } else if (lambda > 0.99 && lambda < 1.01) {
                return(x)
        } else {
                return((x ^ lambda - 1) / lambda)
        }
}
#' @title inv.box.cox
#' @description Runs inverse of Box-Cox transform of a time series
#'
#' @param x Transformed time series object.
#' @param lambda Lambda parameter used for Box-Cox transform.
#' 
#' @return Inverse Box-Cox transformed time series
#' 
#' @author Diego J. Pedregal
#' 
#' @seealso \code{\link{box.cox}}, \code{\link{UC}}, \code{\link{UCforecast}}, \code{\link{UCvalidate}}, \code{\link{UCfilter}}, \code{\link{UCsmooth}}, 
#'          \code{\link{UCdisturb}}, \code{\link{UCcomponents}}
#'          
#' @examples
#' \donttest{
#' y <- inv.box.cox(box.cox(AirPassengers, 0.5), 0.5)
#' plot(y)
#' }
#' @rdname inv.box.cox
#' @export
inv.box.cox <- function(x, lambda){
        if (abs(lambda) < 1e-4){
                return(exp(x))
        } else if (lambda > 0.99 && lambda < 1.01) {
                return(x)
        } else {
                return((x * lambda + 1) ^ (1 / lambda))
        }
}
