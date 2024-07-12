function m = ARIMA(y, s, varargin)
% m = ARIMA(y, s, ...)
%
% Runs all relevant functions for ARIMA modelling
%
% See help of ARIMAmodel.
%
% Author: Diego J. Pedregal
%
% See also: ARIMA, ARIMAmodel, ARIMAvalidate,
%
% Examples
%   m1 = ARIMA(y);
%   m1 = ARIMA(y, 'lambda', []);
%
    % changing verbose
    varargin1 = varargin;
    modeloInd = find(strcmp(varargin1, 'verbose'));
    VERBOSE = false;
    if ~isempty(modeloInd)
        VERBOSE = varargin1{modeloInd + 1};
        varargin1{modeloInd + 1} = false;
    end
    m = ARIMAsetup(y, s, varargin1{:});
    model = m.model;
    m = ARIMAvalidate(m);
    IC = m.IC;
    % deleting model
    modeloInd = find(strcmp(varargin1, 'model'));
    if ~isempty(modeloInd)
      varargin1(modeloInd : modeloInd + 1) = [];
    end
    if model == -99999 && ~m.fast || ~isfinite(IC)
        if s == 1 && sum(abs(m.model(1:3) - [0, 1, 1]')) ~= 0
            % Yearly data
            model = [0, 1, 1, 0, 0, 0];
            m1 = ARIMAsetup(y, s, 'model', model, varargin1{:});
            m1 = ARIMAvalidate(m1);
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
                    m1 = ARIMAvalidate(m1);
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
                m1 = ARIMAvalidate(m1);
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
                    m1 = ARIMAvalidate(m1);
                    if isfinite(m1.IC) && m1.IC < IC
                        IC = m1.IC;
                        m = m1;
                    end
                end
            end
        end
    end
    if VERBOSE
        for i = 1 : length(m.table)
          disp(m.table{i}(1 : end - 1))
        end
    end
end
