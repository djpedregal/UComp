function m1 = ETSmodel(y, s, varargin)
% system = ETSmodel(y, period, 'inp1', inp1, 'inp2', inp2, ...)
%
% ETSmodel is a function for modelling and forecasting univariate time series 
% with ExponenTial Smoothing (ETS) time series models. It sets up the model 
% with a number of control variables that govern the way the rest of functions 
% in the package will work. It also estimates the model parameters by Maximum 
% Likelihood and forecasts the data.
% 
% INPUTS (entered as duplets):
%    y:            a time series to forecast
%    s:            seasonal period of time series (1 for annual, 4 for quarterly, ...)
%    u:            a matrix of input time series. If the output wanted to be forecast, 
%                  matrix u should contain future values for inputs.
%    model:        the model to estimate. It is a single string indicating the type of
%                  model for each component with one or two letters:
%                  Error: ? / A / M
%                  Trend: ? / N / A / Ad / M / Md
%                  Seasonal: ? / N / A / M
%    h:            forecast horizon. If the model includes inputs h is not used, 
%                  the length of u is used instead.
%    criterion:    information criterion for identification ("aic", "bic" or "aicc").
%    lambda:       Box-Cox lambda parameter (NaN for estimation)
%    armaIdent:    check for arma models for error component (true / false).
%    identAll:     run all models to identify the best one (true / false)
%    forIntervals: estimate forecasting intervals (true / false)
%    bootstrap:    use bootstrap simulation for predictive distributions
%    nSimul:       number of simulation runs for bootstrap simulation of predictive 
%                  distributions
%    verbose:      intermediate estimation output (true / false)
%    alphaL:       constraints limits for alpha parameter
%    betaL:        constraints limits for beta parameter
%    gammaL:       constraints limits for gamma parameter
%    phiL:         constraints limits for phi parameter
%
% OUTPUT:
%    system: An object (structure) of class ETS. It is a list with fields including 
%            all the inputs and the fields listed below as outputs. All the functions 
%            in this package fill in part of the fields of any ETS object as 
%            specified in what follows (function ETS fills in all of them at once):
%    
%       After running ETSmodel or ETSestim:
%          p:      Estimated parameters
%          yFor:   Forecasted values of output
%          yForV:  Variance of forecasted values of output
%          ySimul: Bootstrap simulations for forecasting distribution evaluation
% 
%       After running ETSvalidate:
%          table: Estimation and validation table
%          comp:  Estimated components in matrix form
% 
%       After running ETScomponents:
%          comp: Estimated components in matrix form
%
% Author: Diego J. Pedregal
% 
% See also: ETS, ETSvalidate, ETScomponents, ETSestim
% 
% Examples:
%    m = ETSmodel(y, 12);
%    m = ETSmodel(y, 12, 'model', '???');
%    m = ETSmodel(y, 12, 'model', '?AA');                  
    m1 = ETSsetup(y, s, varargin{:});
    m1 = ETSestim(m1);
end