function m = PTScomponents(m)
% system = PTScomponents(system)
%
% Estimates components of PTS models
% 
% INPUTS:
%    system: An object (structure) of class PTS. See help of ETSmodel
%
% OUTPUT:
%    system: The same system as input or a different system with the appropriate 
%            fields filled in
%
% Author: Diego J. Pedregal
% 
% See also: PTS, PTSmodel, PTSvalidate, PTSestim
%
% Examples:
%    m = PTSmodel(y, 12);
%    m = PTScomponents(log(AirPassengers))
    m.modelUCmodel = UCcomponents(m.modelUCmodel);
    m.comp = m.modelUCmodel.comp;
    m.yFit = m.modelUCmodel.yFit;
end