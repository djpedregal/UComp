function m = ETSestim(m)
% system = ETSestim(system)
%
% Estimates and forecasts ETS models
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
% See also: ETS, ETSmodel, ETSvalidate, ETScomponents
%
% Examples:
%    m = ETSsetup(y, 12);
%    m = ETSestim(m)
    model = m.model;
    [m.p, m.model, m.yFor, m.yForV, m.ySimul, m.lambda] = ...
        ETSc('estimate', m.y, m.u, m.model, m.s, m.h, m.criterion, m.armaIdent, ...
             m.identAll, m.forIntervals, m.bootstrap, m.verbose, m.nSimul, m.alphaL, ...
             m.betaL, m.gammaL, m.phiL, m.p0, m.lambda);
    if m.model == "error"
        m.model = model;
        error('Error in function ETSestim.')
    end
end