function m = UCmodel(y,frequency,varargin)
% UCmodel - Estimates and forecasts UC general univariate models
%
%   UCmodel is a function for modelling and forecasting univariate time
%   series according to Unobserved Components models (UC).
%   It sets up the model with a number of control variables that govern the
%   way the rest of functions in the package will work. It also estimates
%   the model parameters by Maximum Likelihood and forecasts the data.
%
%   m = UCmodel(y,frequency)
%   m = UCmodel(y,frequency,'optionalvar1',optvar1,...,'optionalvarN',optvarN)
% 
%   Inputs:
%       y: a time series to forecast.
%       frequency: fundamental period, number of observations per year.
%       periods: (opt) vector of fundamental period and harmonics. If not entered as input, 
%           it will be calculated from frequency.
%       u: (opt) a matrix of external regressors included only in the observation equation. 
%          If the output wanted to be forecast, matrix \code{u} should contain future values 
%          for inputs. Default: []
%       model: (opt) the model to estimate. It is a single string indicating the
%           type of model for each component. It allows two formats
%           'trend/seasonal/irregular' or 'trend/cycle/seasonal/irregular'. The
%           possibilities available for each component are:
%           - Trend: ? / none / rw / irw / llt / dt   
%           - Seasonal: ? / none / equal / different 
%           - Irregular: ? / none / arma(0,0) / arma(p,q) - with p and q
%               integer positive orders
%           - Cycles: ? / none / combination of positive or negative numbers.
%           Positive numbers fix the period of the cycle while negative
%           values estimate the period taking as initial condition the
%           absolute value of the period supplied.
%           Several cycles with positive or negative values are possible
%           and if a question mark is included, the model test for the
%           existence of the cycles specified (check the examples below). 
%           The following are valid cyckle models examples with different 
%           meanings: 48, 48?, -48, -48?, 48+60, -48+60, -48-60, 48-60, 
%                     48+60?, -48+60?, -48-60?, 48-60?.
%           Default: '?/none/?/?'
%       outlier: (opt) critical level of outlier tests. If NaN it does not
%           carry out any outlier detection (default). A negative value
%           indicates critical minimum t test for one run of outlier detection after
%           identification. A positive value indicates the critical
%           minimum t test for outlier detection in any model during identification.
%           Default: NaN
%       stepwise: (opt) stepwise identification procedure (true,false). 
%           Default: false.
%       tTest: (opt) augmented Dickey Fuller test for unit roots (true/false).
%           The number of models to search for is reduced, depending on the
%           result of this test. Default: false
%       p0: (opt) initial condition for parameter estimates. Default: NaN
%       h: (opt) forecast horizon. If the model includes inputs h is not used,
%           the length of u is used instead. Default: NaN
%       criterion: (opt) information criterion for identification ('aic','bic' or
%           'aicc'). Default: 'aic'
%       verbose: (opt) intermediate results shown about progress of estimation
%           (true/false). Default: false.
%       arma: (opt) check for arma models for irregular components (true/false).
%           Default: true
%
%   Output:
%       An object of class UComp. It is a structure with fields including all
%       the inputs and the fields listed below as outputs. All the
%       functions in this package fill in part of the fields of any UComp
%       object as specified in what follows (function UC fills in all of
%       them at once):
%           After running UCmodel or UCestim:
%               p: Estimated parameters
%               v: Estimated innovations (white noise correctly specified
%                   models)
%               yFor: Forecasted values of output
%               yForV: Variance of forecasted values of output
%               criteria: Value of criteria for estimated model
%               iter: Number of iterations in estimation
%               grad: Gradient at estimated parameters
%               covp: Covariance matrix of parameters
% 
%           After running UCvalidate:
%               table: Estimation and validation table
%               v: Estimated innovations (white noise correctly specified models)
%
%           After running UCcomponents:
%               comp: Estimated components in table form
%               compV: Estimated components variance in table form
%             
%           After running UCfilter, UCsmooth or UCdisturb:
%               yFit: Fitted values of output
%               yFitV: Variance of fitted values of output
%               a: State estimates
%               P: Variance of state estimates
%              
%           After running UCdisturb:
%               eta: State perturbations estimates
%               eps: Observed perturbations estimates
%
%   Authors: Diego J. Pedregal, Nerea Urbina
%
%   Examples:
%       load data/airpas
%       m = UCmodel(log(y), 12)
%       m = UCmodel(log(y), 12, 'model', 'llt/equal/arma(0,0)')
%
%   See also UC, UCcomponents, UCdisturb, UCestim, UCfilter, UCsmooth,
%   UCsetup, UCvalidate

    m = UCsetup(y,frequency,varargin{:});
    m = UCestim(m);
    
end