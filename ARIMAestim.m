function m = ARIMAestim(m)
% m = ARIMAestim(m)
%
% ARIMAestim Estimates and forecasts a time series using an ARIMA model
%
% m is an object containing the fields. See help for ARIMAmodel.
%
    if isempty(m.u)
        u = m.u;
    else
        if isvector(m.u)
            u = reshape(m.u, 1, length(m.u));
        else
            [nu1, nu2] = size(m.u);
            u = double(m.u);
            u = reshape(u, nu1, nu2);
        end
    end
    % Call ARIMAc function (assumed to be implemented elsewhere)
    [m.p, yFor, yForV, ySimul, m.lambda, m.model, m.cnst, m.u, m.BIC, ...
        m.AIC, m.AICc, m.IC, m.table, m.v] = ...
             ARIMAc('estimate', double(m.y), u, m.model, m.cnst, m.s, ...
                    m.criterion, m.h, m.verbose, m.lambda, m.maxOrders, ...
                    m.bootstrap, m.nSimul, m.fast, m.identDiff, m.identMethod);
    if isempty(m.p)
        error('Error in ARIMA estimation!!');
    else
        lu = size(m.u, 2); % Ensure the correct dimension is used
        if lu > 0
            m.h = lu - length(m.y);
        end
        if istimetable(m.y) && m.h > 0
            fake = [m.y; NaN]; % Create a fake series with an extra NA
            m.yFor = array2timetable(yFor, 'RowTimes', fake.Properties.RowTimes);
            m.yForV = array2timetable(yForV, 'RowTimes', fake.Properties.RowTimes);
            if m.bootstrap
                m.ySimul = array2timetable(ySimul, 'RowTimes', fake.Properties.RowTimes);
            end
        elseif m.h > 0
            m.yFor = yFor;
            m.yForV = yForV;
            m.ySimul = ySimul;
        end
        return;
    end
end
