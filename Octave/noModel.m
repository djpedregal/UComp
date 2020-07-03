function out = noModel(model,periods)
% noModel - Auxiliar function of UComp library
%
%   out = noModel(model,periods)
% 
%   Inputs:
%       model: reserved input
%       periods: reserved input
%   
%   Author: Diego J. Pedregal

    comps = strsplit(lower(model),'/');
    mT = model(1);
    mC = comps{2}(1);
    if(periods(1) < 2)
        mS = 'n';
    else
        mS = comps{3}(1);
    end
    mI = comps{4}(1);
    if(mT == 'n' && mC == 'n' && mS == 'n' && mI == 'n')
        out = true;
    else
        out = false;
    end
    
end