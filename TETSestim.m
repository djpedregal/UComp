function m = TETSestim(m)
% m = TETSestim(m)
%
% TETTSestim Estimates and forecasts a time series using an TOBIT ETTS model
%
% m is an object containing the fields. See help for TETSmodel.
%
    if isempty(m.u)
        u = m.u;
    else
        if isvector(m.u)
            u = m.u(:)'; % Ensure row vector
        else
            nu = size(m.u);
            u = double(m.u);
            u = reshape(u, nu(1), nu(2));
        end
    end

    [m.p, m.truep, m.model, m.criteria, yFor, yForV, ySimul, m.lambda] = ...
        TETSc('estimate', double(m.y), u, m.model, m.s, m.h, ...
                   m.criterion, m.armaIdent, m.identAll, m.forIntervals, ...
                   m.bootstrap, m.nSimul, m.verbose, m.lambda, ...
                   m.alphaL, m.betaL, m.gammaL, m.phiL, m.p0, m.Ymin, m.Ymax);
    if m.model == "error"
        m.model = model;
        error('Error in function TETSestim.')
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
