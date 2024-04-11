function m = PTSsetup(y, s, varargin)
%     Run PTS general univariate MSOE models
% 
%     Inputs:
%         y:  A time series to forecast (it may be either a numpy vector or a Panda time series object).
%             This is the only input required. If a vector, the additional input 's' should be supplied
%             compulsorily (see below).
%         s: Seasonal period of time series (1 for annual, 4 for quarterly, ...)
%         u:  A matrix of input time series. If the output wanted to be forecast, matrix 'u' should
%             contain future values for inputs.
%         model:  The model to estimate. It is a single string indicating the type of model for each
%                 component with one or two letters:
%                 - Error: ? / N / A
%                 - Trend: ? / N / A / Ad / L
%                 - Seasonal: ? / N / A / D (trigonometric with different variances)
%         h: Forecast horizon. If the model includes inputs, 'h' is not used; the length of 'u' is used instead.
%         criterion: Information criterion for identification ("aic", "bic", or "aicc").
%         lambda: Box-Cox lambda parameter (None: estimate)
%         armaIdent: Check for ARMA models for the error component (True / False).
%         verbose: Intermediate estimation output (True / False)
% 
%     Output:
%         An object of class 'PTS'. It is a structure with fields including all the inputs and the fields
%         listed below as outputs. All the functions in this package fill in part of the fields of any 'PTS'
%         object as specified in what follows (function 'PTS' fills in all of them at once):
% 
%         After running 'PTSmodel':
%         - p0: Initial values for parameter search
%         - p: Estimated parameters
%         - lambdaBoxCox: Estimated Box-Cox lambda parameter
%         - v: Estimated innovations (white noise in correctly specified models)
%         - yFor: Forecasted values of output
%         - yForV: Variance of forecasted values of output
% 
%         After running 'object.validate':
%         - table: Estimation and validation table
% 
%         After running 'object.components':
%         - comp: Estimated components in matrix form
% 
%     See Also:
%         PTS, UCmodel, UC, ETS, ETSmodel
% 
%     Author: Diego J. Pedregal
    menu = inputParser;
    addRequired(menu, 'y', @isfloat);
    addRequired(menu, 's', @isfloat);
    addParameter(menu, 'u', [], @isfloat);
    addParameter(menu, 'model', '???', @ischar);
    addParameter(menu, 'h', 24, @isfloat);
    addParameter(menu, 'criterion', 'aicc', @ischar);
    addParameter(menu, 'lambda', 1.0, @isfloat);
    addParameter(menu, 'armaIdent', false, @islogical);
    addParameter(menu, 'verbose', false, @islogical);
    parse(menu, y, s, varargin{:});
    % Checking values
    lambda = menu.Results.lambda;
    if isnan(lambda)
        lambda = 9999.9;
    end
    m  = struct('y', menu.Results.y, ...
                'u', menu.Results.u, ...
                'model', menu.Results.model, ...
                's', menu.Results.s, ...
                'h', menu.Results.h, ...
                'p0', [], ...
                'criterion', menu.Results.criterion, ...
                'lambda', lambda, ...
                'armaIdent', menu.Results.armaIdent, ...
                'verbose', menu.Results.verbose, ...
                'armaOrders', [0 0], ...
                'yFor', [], ...
                'yForV', [], ...
                'comp', [], ...
                'yFit', [], ...
                'table', '', ...
                'p', [], ...
                'v', [], ...
                'modelUC', '', ...
                'modelUCmodel', [], ...
                'periods', []);
    if s < 2
        m.periods = 1;
    else
        m.periods = s ./ (1 : floor(s / 2));
    end
    m.modelUC = PTS2modelUC(m.model, m.armaOrders);
    m.modelUCmodel = UCsetup(y, s, 'trendOptions', 'rw/llt/srw', ...
                             'seasonalOptions', 'none/linear/different', ...
                             'irregularOptions', 'none/arma(0,0)', ...
                             'MSOE', true, 'PTSnames', true, varargin{:}, ...
                             'model', m.modelUC);
end
