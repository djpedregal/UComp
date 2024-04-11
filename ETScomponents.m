function m = ETScomponents(m)
% system = ETScomponents(system)
%
% Estimates components of ETS models
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
% See also: ETS, ETSmodel, ETSvalidate, ETSestim
%
% Examples:
%    m = ETSmodel(y, 12);
%    m = ETScomponents(log(AirPassengers))
    [comp, compNames] = ETSc('components', ...
        m.y, m.u, m.model, m.s, m.h, m.criterion, m.armaIdent, m.identAll, m.forIntervals, m.bootstrap, ...
        m.verbose, m.nSimul, m.alphaL, m.betaL, m.gammaL, m.phiL, m.p0, m.lambda);
    m.v = comp(:, 1);
    if ~exist ('OCTAVE_VERSION', 'builtin')   % Matlab
        m.comp = array2table(comp, 'VariableNames', split(string(compNames), "/"));
    end
end
    