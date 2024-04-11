function m = ETSsetup(y, s, varargin)
% system = ETSsetup(y, period, 'inp1', inp1, 'inp2', inp2, ...)
%
% Sets up ETS general univariate models
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
%    system: An object (structure) of class ETS. See help of ETSmodel
%    
% Author: Diego J. Pedregal
% 
% See also: ETS, ETSmodel, ETSvalidate, ETScomponents, ETSestim
% 
% Examples:
%    m = ETSsetup(y, 12);
%    m = ETSsetup(y, 12, 'model', '???');
%    m = ETSsetup(y, 12, 'model', '?AA');
    menu = inputParser;
    addRequired(menu, 'y', @isfloat);
    addRequired(menu, 's', @isfloat);
    addParameter(menu, 'u', -99999, @isfloat);
    addParameter(menu, 'model', '???', @ischar);
    addParameter(menu, 'h', 24, @isfloat);
    addParameter(menu, 'criterion', 'aicc', @ischar);
    addParameter(menu, 'armaIdent', false, @islogical);
    addParameter(menu, 'identAll', false, @islogical);
    addParameter(menu, 'forIntervals', false, @islogical);
    addParameter(menu, 'bootstrap', false, @islogical);
    addParameter(menu, 'verbose', false, @islogical);
    addParameter(menu, 'nSimul', 5000, @isfloat);
    addParameter(menu, 'alphaL', [0 1], @isfloat);
    addParameter(menu, 'betaL', [0 1], @isfloat);
    addParameter(menu, 'gammaL', [0 1], @isfloat);
    addParameter(menu, 'phiL', [0.8 0.98], @isfloat);
    addParameter(menu, 'p0', -99999, @isfloat);
    addParameter(menu, 'lambda', 1.0, @isfloat);
    parse(menu, y, s, varargin{:});
    % Checking values
    alphaL = menu.Results.alphaL;
    betaL = menu.Results.betaL;
    gammaL = menu.Results.gammaL;
    phiL = menu.Results.phiL;
    if (alphaL(1) >= alphaL(2))
        error("Wrong alpha limits!!")
    end
    if (betaL(1) >= betaL(2))
        error("Wrong alpha limits!!")
    end
    if (gammaL(1) >= gammaL(2))
        error("Wrong alpha limits!!")
    end
    if (phiL(1) >= phiL(2))
        error("Wrong alpha limits!!")
    end
    lambda = menu.Results.lambda;
    if isnan(lambda)
        lambda = 9999.9;
    end
    m  = struct('y', menu.Results.y, ...
                'u', menu.Results.u, ...
                'model', menu.Results.model, ...
                's', menu.Results.s, ...
                'h', menu.Results.h, ...
                'p0', menu.Results.p0, ...
                'criterion', menu.Results.criterion, ...
                'lambda', lambda, ...
                'armaIdent', menu.Results.armaIdent, ...
                'identAll', menu.Results.identAll, ...
                'forIntervals', menu.Results.forIntervals, ...
                'bootstrap', menu.Results.bootstrap, ...
                'nSimul', menu.Results.nSimul, ...
                'verbose', menu.Results.verbose, ...
                'alphaL', menu.Results.alphaL, ...
                'betaL', menu.Results.betaL, ...
                'gammaL', menu.Results.gammaL, ...
                'phiL', menu.Results.phiL, ...
                'yFor', [], ...
                'yForV', [], ...
                'comp', [], ...
                'ySimul', [], ...
                'table', '', ...
                'p', []);
end
