function m1 = TETSmodel(y, s, varargin)
% system = TETSmodel(y, period, 'inp1', inp1, 'inp2', inp2, ...)
%
% Estimates and forecasts TOBIT ETS general univariate models
% 
% INPUTS (entered as duplets):
%    y:            a time series to forecast
%    period:       seasonal period of time series (1 for annual, 4 for quarterly, ...)
%    u:            a matrix of input time series. If the output wanted to be forecast, 
%                  matrix u should contain future values for inputs.
%    model:        the model to estimate. It is a single string indicating the type of
%                  model for each component with one or two letters:
%                  Error: ? / A
%                  Trend: ? / N / A / Ad
%                  Seasonal: ? / N / A
%    h:            forecast horizon. If the model includes inputs h is not used, 
%                  the length of u is used instead.
%    criterion:    information criterion for identification ("aic", "bic" or "aicc").
%    forIntervals: estimate forecasting intervals (true / false)
%    bootstrap:    use bootstrap simulation for predictive distributions
%    nSimul:       number of simulation runs for bootstrap simulation of predictive 
%                  distributions
%    verbose:      intermediate estimation output (true / false)
%    alphaL:       constraints limits for alpha parameter
%    betaL:        constraints limits for beta parameter
%    gammaL:       constraints limits for gamma parameter
%    phiL:         constraints limits for phi parameter
%    Ymin:         scalar or vector of time varying censoring values from below (default -Inf)
%    Ymax:         scalar or vector of time varying censoring values from above (default Inf)
%
% OUTPUT:
%    system: An object (structure) of class TETS. It is a list with fields including 
%            all the inputs and the fields listed below as outputs. All the functions 
%            in this package fill in part of the fields of any TETS object as 
%            specified in what follows (function ETS fills in all of them at once):
%    
%       After running TETSmodel or TETSestim:
%          p:      Estimated parameters
%          yFor:   Forecasted values of output
%          yForV:  Variance of forecasted values of output
%          ySimul: Bootstrap simulations for forecasting distribution evaluation
% 
%       After running TETSvalidate:
%          table: Estimation and validation table
%          comp:  Estimated components in matrix form
% 
%       After running TETScomponents:
%          comp: Estimated components in matrix form
%    
% Author: Diego J. Pedregal
% 
% See also: TETS, TETSsetup, TETSvalidate, TETScomponents, TETSestim
% 
% Examples:
%    m = TETSmodel(y, 12);
%    m = TETSmodel(y, 12, 'model', '???');
%    m = TETSmodel(y, 12, 'model', '?AA');
    m1 = TETSsetup(y, s, varargin{:});
    if (min(Ymax -y, [], 'omitnan') > 0 && min(y - Ymin, [], 'omitnan') > 0)
        m1 = ETSestim(m1);
    else
        m1 = TETSestim(m1);
    end
end
