function sys = UCdisturb(sys)
% UCdisturb - Runs the Disturbance Smoother for UC models
%    
%   sys = UCdisturb(sys)   
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
%           eta:   State perturbations estimates
%           eps:   Observed perturbations estimates        
%
%   Authors: Diego J. Pedregal, Nerea Urbina
%    
%   Examples:
%       load data/airpas
%       m = UCmodel(log(y), 12)
%       m = UCdisturb(m)
%        
%   See also UC, UCcomponents, UCsmooth, UCestim, UCfilter, UCmodel,
%   UCsetup, UCvalidate

    sys = filter_(sys,'disturb');

end