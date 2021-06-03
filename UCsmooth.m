function sys = UCsmooth(sys)
% UCsmooth  - Runs the Fixed Interval Smoother for UC models
% 
%   sys = UCsmooth(sys)
%   
%   Input: 
%       sys: structure of type UComp created with UCmodel
%   
%   Output:
%       Returns the same input structure with the appropiate fields filled in,
%       in particular:
%           yFit:  Fitted values of output
%           yFitV: Variance of fitted values of output
%           a:     State estimates
%           P:     Variance of state estimates (diagonal of covariance matrices)
%
%   Authors: Diego J. Pedregal, Nerea Urbina
%
%   Examples:
%       load data/airpas
%       m = UCmodel(log(y), 12)
%       m = UCsmooth(m)
%
%   See also UC, UCcomponents, UCdisturb, UCestim, UCfilter, UCmodel,
%   UCsetup, UCvalidate

    sys = filter_(sys,'smooth');
    
end