# UComp
UComp is a library that implements comprehensive procedures of identification, estimation and forecasting of time series models based on a number of techniques, some well-known, others exclusive of this toolbox. All models are univarite:
 - Unobserved Components (UC) in the spirit of Harvey (1989). The feature that makes UComp unique among competitors is that models may be automatically identified, without any human intervention, by information criteria. This means that the user do not need to impose any prior structure onthe model, because it may be decided by UComp. This algorithm is implemented with total flexibility, in the sense that the user may decide whether to let it pick all the components, one or part of them. There are several options for the search, one running the whole population of models or just doing some stepwise procedures.
- ExponenTial Smoothing (ETS) in the spirit of Hyndman et al. (2008).
- ARIMA. Identification is extraordinarily fast, because Hannan-Rissanen algorithm is used.
- TETS: Tobit (or censored) ExponenTial Smoothing with constraints from above and below. It solves this problem in an elegant manner. It also includes automatic identification.
- PTS (experimental). Multiple source of error ExponenTial Smoothing. Trying to make the ETS approach more flexible. Auotmatic identification provided.

A very important issue is that UComp is fully coded in C++. This ensures optimal execution speed and the possibility to 'link' it to many popular environments by just writing the appropriate wrapper functions. At the moment there are versions written in R (installable from CRAN repository, [here](https://cran.r-project.org/web/packages/UComp/index.html)), MATLAB and Octave in this repository, and Python (in PyPI repository).

## Installation: 

UComp is written in C++. To make it functional in MATLAB you need to compile the toolbox by means of the function mexUComp.m. You need to download Armadillo C++ library and have access to Lapack and BLAS libraries. Depending on your particular configuration, such libraries may be already installed. Otherwise, proper installation would be needed prior to installing UComp. More details in section 3.2 of the manual and in Readme.txt. Editing mexUComp.m can also be helpful. In most systems mexUComp only needs the first input that tells the folder where Armadillo library is located.


## References: 

Harvey, AC (1989). Forecasting, structural time series models and the Kalman filter. Cambridge university press.

de Jong, P. & Penzer, J. (1998). Diagnosing Shocks in Time Series, Journal of the American Statistical Association, 93, 442, 796-806.

Pedregal, D. J., & Young, P. C. (2002). Statistical approaches to modelling and forecasting time series. In M. Clements, & D. Hendry (Eds.), Companion to economic forecasting (pp. 69–104). Oxford: Blackwell Publishers.

Durbin J, Koopman SJ (2012). Time Series Analysis by State Space Methods. 38. Oxford University Press.

Proietti T. and Luati A. (2013). Maximum likelihood estimation of time series models: the Kalman filter and beyond, in Handbook of research methods and applications in empirical macroeconomics, ed. Nigar Hashimzade and Michael Thornton, E. Elgar, UK.

## Everybody welcome

Any ideas, comments and improvements are welcome (send me an email)!!


