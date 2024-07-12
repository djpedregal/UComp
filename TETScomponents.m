function m = TETScomponents(m)
% m = TETScomponents(m)
%
% TETScomponents Shows a table of estimation and diagnostics results for TOBIT ETS models
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

    [comp, compNames] ...
        = TETSc('components', double(m.y), u, m.model, m.s, m.h, ...
                   m.criterion, m.armaIdent, m.identAll, m.forIntervals, ...
                   m.bootstrap, m.nSimul, m.verbose, m.lambda, ...
                   m.alphaL, m.betaL, m.gammaL, m.phiL, m.p0, m.Ymin, m.Ymax);

    if isdatetime(m.y)
        m.comp = array2timetable(output.comp, 'RowTimes', m.y);
    else
        m.comp = comp;
    end

    % Convert component names from '/' separated string to cell array of strings
    compNames = strsplit(compNames, '/');
    m.comp.Properties.VariableNames = compNames;

    return;
end
