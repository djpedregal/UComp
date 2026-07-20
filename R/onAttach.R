#.First.lib <-
.onAttach <- function(libname, pkgname) {
    ver <- packageDescription("UComp")$Version
    txt <- c("\n",
             paste(sQuote("UComp"), "version:", ver),
             paste(sQuote("UComp"),
                   "is a package for time series modelling"),
             "using Unobserved Components, Exponential ",
             "smoothing, ARIMA models and Tobit ETS",
             "\n",
             "Author: Diego J. Pedregal",
             "Diego.Pedregal@uclm.es",
             "\n"
    )
    if (interactive() || getOption("verbose")) {
        msg = paste(strwrap(txt, indent = 4, exdent = 4), collapse = "\n")
        packageStartupMessage(msg)
    }
}
