# UComp
UComp is a library that implements comprehensive procedures of identification, estimation and forecasting of time series models based on univariate Unobserved Components models (UC) in the spirit of Harvey (1989). The feature that makes UComp unique among competitors is that models may be automatically identified, without any human intervention, by information criteria. This means that the user do not need to impose any prior structure onthe model, because it may be decided by UComp. This algorithm is implemented with total flexibility, in the sense that the user may decide whether to let it pick all thecomponents, one or part of them. There are several options for the search, one running the whole population of models or just doing some stepwise procedures,

Another important issue is that it is fully coded in C++. This ensures optimal execution speed and the possibility to 'link' it to many popular environments by just writing the appropriate wrapper functions. At the moment there are versions written in R (installable from CRAN repository, [here](https://cran.r-project.org/web/packages/UComp/index.html)), and MATLAB and Octave in this repository.

**POTENTIAL CONTRIBUTORS**: At the moment, I am looking for contributors that might be interested on 'linking' it to other environments, like Python, Go, Julia, and/or any other... The main difficulty, I guess, is the need translate the new environment datatypes to and from Armaillo, that is the linear algebra in which UComp is written. 

Any ideas and improvements are welcome!!

Reference: Harvey, AC (1989). Forecasting, structural time series models and the Kalman filter. Cambridge university press.
