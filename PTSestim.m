function m = PTSestim(m)
% system = PTSestim(system)
%
% Estimates and forecasts PTS models
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
% See also: PTS, PTSmodel, PTSvalidate, PTScomponents
%
% Examples:
%    m = PTSsetup(y, 12);
%    m = PTSestim(m)end    
    model = m.model;
    m.modelUCmodel = UCestim(m.modelUCmodel);
    m.model = modelUC2PTS(m.modelUCmodel.model);
    m.p0 = m.modelUCmodel.p0;
    m.lambda = m.modelUCmodel.lambda;
    m.armaOrders = m.modelUCmodel.arma;
    m.yFor = m.modelUCmodel.yFor;
    m.yForV = m.modelUCmodel.yForV;
    m.periods = m.modelUCmodel.periods;
    m.p = m.modelUCmodel.p;
    if m.model == "error"
        m.model = model;
        error('Error in function PTSestim.')
    end
end
