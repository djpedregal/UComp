function m = ARIMAmodel(y, s, varargin)
% m = ARIMAmodel(y, s, 'inp1', inp1, 'inp2', inp2, ...)
%
% Estimates and forecasts ARIMA general univariate models
%
% ARIMAmodel is a function for modelling and forecasting univariate
% time series with Autoregressive Integrated Moving Average (ARIMA) time series models.
% It sets up the model with a number of control variables that
% govern the way the rest of functions in the package will work. It also estimates
% the model parameters by Maximum Likelihood and forecasts the data.
%
% INPUTS:
%   y:          a time series to forecast (it may be either a numerical vector or
%               a time series object). This is the only input required. If a vector, the additional
%               input s should be supplied compulsorily (see below).
%   s:          seasonal period of time series (1 for annual, 4 for quarterly, ...)
%   u:          a matrix of input time series. If
%               the output wanted to be forecast, matrix u should contain future values for inputs.
%   model:      the model to estimate. A vector [p,d,q,P,D,Q] containing the model orders
%               of an ARIMA(p,d,q)x(P,D,Q)_s model. A constant may be estimated with the
%               cnst input.
%               Use an empty matrix to automatically identify the ARIMA model.
%   cnst:       flag to include a constant in the model (1/0/NaN). Use NaN to estimate
%   h:          forecast horizon. If the model includes inputs h is not used, the length of u is used instead.
%   criterion:  information criterion for identification stage ("aic", "bic", "aicc")
%   verbose:    intermediate estimation output (TRUE / FALSE)
%   lambda:     Box-Cox lambda parameter (NULL: estimate)
%   maxOrders:  a vector [p,d,q,P,D,Q] containing the maximum orders of model orders
%               to search for in the automatic identification
%   bootstrap:  use bootstrap simulation for predictive distributions
%   nSimul:     number of simulation runs for bootstrap simulation of predictive distributions
%   fast:       fast identification (avoids post-identification checks)
%
% OUTPUT:
%    system: An object of class ARIMA. It is a list with fields including all the inputs and
%            the fields listed below as outputs. All the functions in this package fill in
%            part of the fields of any ARIMA object as specified in what follows (function
%            ARIMA fills in all of them at once):
%
% After running ARIMAmodel or ARIMA:
% p:        Estimated parameters
% yFor:     Forecasted values of output
% yForV:    Variance of forecasted values of output
% ySimul:   Bootstrap simulations for forecasting distribution evaluation
%
% After running ARIMAvalidate:
% table:    Estimation and validation table
%
% Author: Diego J. Pedregal
%
% See also: ARIMA, ARIMAvalidate,
%
% @examples
% m1 = ARIMAmodel(y);
% m1 = ARIMAmodel(y, 'lambda', []);
%
    m = ARIMAsetup(y, s, varargin{:});
    model = m.model;
    m = ARIMAestim(m);
    IC = m.IC;
    varargin1 = varargin;
    % deleting model
    modeloInd = find(strcmp(varargin1, 'model'));
    if ~isempty(modeloInd)
      varargin1(modeloInd : modeloInd + 1) = [];
    end
    % changing verbose
    modeloInd = find(strcmp(varargin1, 'verbose'));
    if ~isempty(modeloInd)
      varargin1{modeloInd + 1} = false;
    end
    if model == -99999 && ~m.fast || ~isfinite(IC)
        if s == 1 && sum(abs(m.model(1:3) - [0, 1, 1]')) ~= 0
            % Yearly data
            model = [0, 1, 1, 0, 0, 0];
            m1 = ARIMAsetup(y, s, 'model', model, varargin1{:});
            m1 = ARIMAestim(m1);
            if isfinite(m1.IC) && m1.IC < IC
                IC = m1.IC;
                m = m1;
            end
            if m.model(2) > 0
                model = m.model;
                model(2) = model(2) - 1;
                model(3) = min(model(3) + 1, m.maxOrders(3));
                if ~all(model == [0, 1, 1, 0, 0, 0]')
                    m1 = ARIMAsetup(y, s, 'model', model, varargin1{:});
                    m1 = ARIMAestim(m1);
                    if isfinite(m1.IC) && m1.IC < IC
                        IC = m1.IC;
                        m = m1;
                    end
                end
            end
        elseif s > 1
            % Non-yearly data
            if sum(abs(m.model - [0, 1, 1, 0, 1, 1]')) ~= 0
                model = [0, 1, 1, 0, 1, 1];
                m1 = ARIMAsetup(y, s, 'model', model, varargin1{:});
                m1 = ARIMAestim(m1);
                if isfinite(m1.IC) && m1.IC < IC
                    IC = m1.IC;
                    m = m1;
                end
            end
            if m.model(2) > 0 && m.model(5) > 0
                model = m.model;
                model(2) = model(2) - 1;
                model(1) = min(model(1) + 1, m.maxOrders(1));
                if ~all(model == [0, 1, 1, 0, 1, 1]')
                    m1 = ARIMAsetup(y, s, 'model', model, varargin1{:});
                    m1 = ARIMAestim(m1);
                    if isfinite(m1.IC) && m1.IC < IC
                        IC = m1.IC;
                        m = m1;
                    end
                end
            end
        end
    end
end
