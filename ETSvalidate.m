function m = ETSvalidate(m, show)
% system = ETSvalidate(system)
%
% Shows a table of estimation and diagnostics results for UC models
% 
% INPUTS:
%    system: An object (structure) of class ETS. See help of ETSmodel
%
% OUTPUT:
%    system: The same system as input or a different system with the appropriate 
%            fields filled in
%
% Author: Diego J. Pedregal
% 
% See also: ETS, ETSmodel, ETScomponents, ETSestim
%
% Examples:
%    m = ETSmodel(y, 12);
%    m = ETSvalidate(log(AirPassengers))
    if nargin < 2
        show = true;
    end
    [comp, table, compNames] = ETSc('validate', ...
        m.y, m.u, m.model, m.s, m.h, m.criterion, m.armaIdent, m.identAll, m.forIntervals, m.bootstrap, ...
        false, m.nSimul, m.alphaL, m.betaL, m.gammaL, m.phiL, m.p0, m.lambda);
    m.v = comp(:, 1);
    if ~exist ('OCTAVE_VERSION', 'builtin')   % Matlab
        m.comp = array2table(comp, 'VariableNames', split(string(compNames), "/"));
    end
    m.table = table;
    if show
        for i = 1 : length(m.table)
          disp(m.table{i}(1 : end - 1))
        end
    end

end
    