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
    [m.p, m.model, yFor, yForV, ySimul, m.lambda] = ...
        ETSc('estimate', m.y, m.u, m.model, m.s, m.h, m.criterion, m.armaIdent, ...
             m.identAll, m.forIntervals, m.bootstrap, m.verbose, m.nSimul, m.alphaL, ...
             m.betaL, m.gammaL, m.phiL, m.p0, m.lambda);
    if m.model == "error"
        m.model = model;
        error('Error in function ETSestim.')
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