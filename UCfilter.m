function sys = UCfilter(sys)
% UCfilter - Runs the Kalman Filter for UC models
%    
%   sys = UCfilter(sys)   
%
%   Input:
%       sys: structure of type UComp created with UCmodel
%    
%   Output:
%       The same input structure with the appropiate fields filled in, in particular:
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
%       m = UCfilter(m)
%        
%   See also UC, UCcomponents, UCdisturb, UCestim, UCsmooth, UCmodel,
%   UCsetup, UCvalidate

    sys = filter_(sys,'filter');
    
end