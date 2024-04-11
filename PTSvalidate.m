function m = PTSvalidate(m, show)
% system = PTSvalidate(system)
%
% Shows a table of estimation and diagnostics results for PTS models
% 
% INPUTS:
%    system: An object (structure) of class PTS. See help of PTSmodel
%
% OUTPUT:
%    system: The same system as input or a different system with the appropriate 
%            fields filled in
%
% Author: Diego J. Pedregal
% 
% See also: PTS, PTSmodel, PTScomponents, PTSestim
%
% Examples:
%    m = PTSmodel(y, 12);
%    m = PTSvalidate(log(AirPassengers))
    if nargin < 2
        show = true;
    end
    m.modelUCmodel = UCvalidate(m.modelUCmodel, show);
    m.table = m.modelUCmodel.table;
    m.v = m.modelUCmodel.v;
    m.yFit = m.modelUCmodel.yFit;
end
    