function m = TETSvalidate(m)
% m = TETSvalidate(m)
%
% TETSvalidate Shows a table of estimation and diagnostics results for TOBIT ETS models
%
% m is an object containing the fields. See help for TETSmodel.
%
    if isempty(m.u)
        u = m.u;
    else
        if isvector(m.u)
            u = reshape(m.u, 1, numel(m.u));
        else
            nu = size(m.u);
            u = reshape(m.u, nu(1), nu(2));
        end
    end
    
    [comp, table, compNames] = ...
        TETSc('validate', double(m.y), u, m.model, m.s, m.h, ...
                   m.criterion, m.armaIdent, m.identAll, m.forIntervals, ...
                   m.bootstrap, m.nSimul, m.verbose, m.lambda, ...
                   m.alphaL, m.betaL, m.gammaL, m.phiL, m.p0, m.Ymin, m.Ymax);
    m.v = comp(:, 1);
    if ~exist ('OCTAVE_VERSION', 'builtin')   % Matlab
        m.comp = array2table(comp, 'VariableNames', split(string(compNames), "/"));
        if istimetable(m.y)
            m.comp = timetable(m.comp, 'Start', m.y.TimeInfo.Start, 'SampleRate', m.y.TimeInfo.SampleRate);
        end
    else
        m.comp = comp;
    end
    m.table = table;
    % Handling NaN p-values in table
    ind = contains(m.table, 'nan');
    if any(ind)
        for i = 1:length(ind)
            line = m.table{ind(i)};
            df = str2double(line(9:12));
            Fstat = str2double(line(15:31));
            pval = round(fcdf(Fstat, df, df), 4);
            line = replaceBetween(line, 9, 12, num2str(pval));
            m.table{ind(i)} = line;
        end
    end
    % if m.verbose
    %     for i = 1 : length(m.table)
    %       disp(m.table{i}(1 : end - 1))
    %     end
    % end
end
