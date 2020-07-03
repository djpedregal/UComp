function m = UC(y,frequency,varargin)
% UC - Runs all relevant functions for UC modelling
%
%   See help of UCsetup
%
%   m = UC(y,frequency)
%   m = UC(y,frequency,'optionalvar1',optvar1,...,'optionalvarN',optvarN)
%
%   Inputs:
%       y: a time series to forecast.
%       frequency: fundamental period, number of observations per year.
%       periods: (opt) vector of fundamental period and harmonics. If not entered as input, 
%           it will be calculated from frequency.
%       u: (opt) a matrix of input time series. If the output wanted to be
%           forecast, matrix u should contain future values of inputs.
%           Default: []
%       model: (opt) the model to estimate. It is a single string indicating the
%           type of model for each component. It allows two formats
%           'trend/seasonal/irregular' or 'trend/cycle/seasonal/irregular'. The
%           possibilities available for each component are:
%           - Trend: ? / none / rw / irw / llt / dt   
%           - Seasonal: ? / none / equal / different 
%           - Irregular: ? / none / arma(0,0) / arma(p,q) - with p and q
%               integer positive orders
%           - Cycles: ? / none / combination of positive or negative numbers 
%           Positive numbers fix the period of the cycle while negative
%           values estimare the period taking as initial condition the
%           absolute value of the period supplied.
%           Several cycles with positive or negative values are possible
%           and if a question mark is included, the model test for the
%           existence of the cycles specified (check the examples below).
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
%           the length of u is used intead. Default: NaN
%       criterion: (opt) information criterion for identification ('aic','bic' or
%           'aicc'). Default: 'aic'
%       verbose: (opt) intermediate results shown about progress of estimation
%           (true/false). Default: false.
%       arma: (opt) check for arma models for irregular components (true/false).
%           Default: true
%       cLlik: (opt) reserved input
%
%   Output:
%       An object of class UComp. It is a structure with fields including all
%       the inputs and the fields listed below as outputs:
%
%           After running UCestim:
%               p: Estimated parameters
%               v: Estimated innovations (white noise correctly specified
%                   models)
%               yFor: Forecasted values of output
%               yForV: Variance of forecasted values of output
%               criteria: Value of criteria for estimated model
% 
%           After running UCdisturb:
%               yFit: Fitted values of output
%               yFitV: Variance of fitted values of output
%               a: State estimates
%               P: Variance of state estimates
%               eta: State perturbations estimates
%               eps: Observed perturbations estimates
%
%           After running UCvalidate:
%               table: Estimation and validation table
%                 
%           After running UCcomponents:
%               comp: Estimated components in table form
%               compV: Estimated components variance in table form
%             
%   Authors: Diego J. Pedregal, Nerea Urbina
%
%   Examples:
%       load 'airpassengers' - contains 2 variables: y, frequency
%       m = UC(log(y),frequency)
%       m = UC(log(y),frequency,'model','llt/equal/arma(0,0)')
%
%   See also UCsmooth, UCcomponents, UCdisturb, UCestim, UCfilter, UCmodel,
%   UCsetup, UCvalidate

    m = UCsetup(y,frequency,varargin{:});
    m = UCestim(m);
    m = UCdisturb(m);
    m = UCvalidate(m);
    m = UCcomponents(m);
    
end