function m1 = ETS(y, s, varargin)
% system = ETS(y, period, 'inp1', inp1, 'inp2', inp2, ...)
%
% Runs all relevant functions for ETS modelling. See details in help of
% ETSmodel
% 
% INPUTS (entered as duplets):
%    y:            a time series to forecast
%    period:       seasonal period of time series (1 for annual, 4 for quarterly, ...)
%    u:            a matrix of input time series. If the output wanted to be forecast, 
%                  matrix u should contain future values for inputs.
%    model:        the model to estimate. It is a single string indicating the type of
%                  model for each component with one or two letters:
%                  Error: ? / A / M
%                  Trend: ? / N / A / Ad / M / Md
%                  Seasonal: ? / N / A / M
%    h:            forecasting horizon. If the model includes inputs h is not used, 
%                  the length of u is used instead.
%    criterion:    information criterion for identification ("aic", "bic" or "aicc").
%    lambda:       Box-Cox lambda transformation coefficient (NaN for estimation)
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
%    system: An object (structure) of class ETS. See help of ETSmodel
%    
% Author: Diego J. Pedregal
% 
% See also: ETSmodel, ETSvalidate, ETScomponents, ETSestim
% 
% Examples:
%    m = ETS(y, 12);
%    m = ETS(y, 12, 'model', '???');
%    m = ETS(y, 12, 'model', 'MAM', 'bootstrap', true);
    m1 = ETSsetup(y, s, varargin{:});
    m1 = ETSestim(m1);
    m1 = ETSvalidate(m1, m1.verbose);
end